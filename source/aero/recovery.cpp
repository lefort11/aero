#include "recovery.h"
#include "startup.h"
#include "../io/lzma_stream.h"
#include "../io/sha256.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace aero {
namespace {
using Id=std::uint64_t;
const unsigned header_size=120;
using Packet=std::array<char,header_size>;
void require(bool ok,const std::string& message) {
    if (!ok) throw std::runtime_error("recovery: "+message);
}
void put(char* dst,Id value) {
    for (unsigned i=0;i<8;++i) dst[i]=static_cast<char>(value>>(8*i));
}
Id get(const char* src) {
    Id value=0;
    for (unsigned i=0;i<8;++i) value|=Id(static_cast<unsigned char>(src[i]))<<(8*i);
    return value;
}
using indexed_io::put_real;
double real(const char* src) { const Id bits=get(src); double value; std::memcpy(&value,&bits,8); return value; }
unsigned record_size(unsigned width) { return 24+16*width; }
std::streamoff offset(Id id,unsigned width) {
    const auto stride=record_size(width);
    require(id<=(static_cast<Id>(std::numeric_limits<std::streamoff>::max())-header_size)/stride,"file too large");
    return static_cast<std::streamoff>(header_size+id*stride);
}
void seal(char* bytes,unsigned size) { put(bytes+size-8,io::crc32(bytes,size-8)); }
bool intact(const char* bytes,unsigned size) {
    return get(bytes+size-8)==io::crc32(bytes,size-8);
}
Packet header(Id count,unsigned width,double gamma,const std::string& hash,Id step,double time) {
    Packet bytes{};
    std::memcpy(bytes.data(),"AERORCV1",8);
    put(&bytes[8],width-2); put(&bytes[16],count); put_real(&bytes[24],gamma);
    put(&bytes[32],step); put_real(&bytes[40],time);
    std::memcpy(&bytes[48],hash.data(),64);
    seal(bytes.data(),header_size);
    return bytes;
}
Packet read_header(std::ifstream& in,Id count,unsigned width,double gamma,const std::string& hash) {
    Packet bytes{};
    in.read(bytes.data(),header_size);
    require(in && !std::memcmp(bytes.data(),"AERORCV1",8) && intact(bytes.data(),header_size),"incomplete or damaged header");
    require(get(&bytes[8])==width-2 && get(&bytes[16])==count && real(&bytes[24])==gamma &&
            !std::memcmp(&bytes[48],hash.data(),64),"incompatible mesh or gas");
    require(get(&bytes[32]) && std::isfinite(real(&bytes[40])) && real(&bytes[40])>0,"invalid step or time");
    in.seekg(0,std::ios::end);
    require(in.tellg()==offset(count,width),"incomplete or damaged file length");
    return bytes;
}
bool exists(const std::string& path) {
    struct stat info;
    if (!stat(path.c_str(),&info)) return true;
    require(errno==ENOENT,"cannot inspect "+path);
    return false;
}
void sync_file(const std::string& path) {
#ifdef _WIN32
    HANDLE f=CreateFileA(path.c_str(),GENERIC_WRITE,FILE_SHARE_READ|FILE_SHARE_WRITE,nullptr,OPEN_EXISTING,0,nullptr);
    require(f!=INVALID_HANDLE_VALUE,"cannot open for flush: "+path);
    const bool ok=FlushFileBuffers(f)!=0;
    CloseHandle(f);
#else
    const int f=open(path.c_str(),O_RDWR);
    require(f>=0,"cannot open for flush: "+path);
    const bool ok=fsync(f)==0;
    close(f);
#endif
    require(ok,"cannot flush "+path);
}
// Old archives have no native mesh digest. Bind those checkpoints to the
// exact archive bytes. Only rank zero reads this stream, once per launch.
std::string archive_hash(const std::string& path,const Processes& world) {
    int digits[64]={};
    world.local([&] {
        if (world.rank) return;
        std::ifstream in(path,std::ios::binary);
        require(in.is_open(),"cannot open "+path);
        io_detail::Sha256Buffer hash;
        char block[65536];
        while (in.read(block,sizeof block) || in.gcount()) hash.sputn(block,in.gcount());
        require(in.eof() && !in.bad(),"cannot read "+path);
        const auto digest=hash.hex_digest();
        std::copy(digest.begin(),digest.end(),digits);
    });
    world.broadcast(digits,64,0);
    std::string hash(64,0);
    for (unsigned i=0;i<hash.size();++i) hash[i]=static_cast<char>(digits[i]);
    return hash;
}
}
// The OS releases the writer lock when the owning process exits, even on a crash.
struct Recovery::Lock {
#ifdef _WIN32
    HANDLE handle;
    explicit Lock(const std::string& path) {
        handle=CreateFileA(path.c_str(),GENERIC_WRITE,0,nullptr,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr);
        require(handle!=INVALID_HANDLE_VALUE,"cannot lock "+path);
    }
    ~Lock() { CloseHandle(handle); }
#else
    int fd;
    explicit Lock(const std::string& path) {
        fd=open(path.c_str(),O_RDWR|O_CREAT,0666);
        require(fd>=0,"cannot open "+path);
        if (flock(fd,LOCK_EX|LOCK_NB)) { close(fd); throw std::runtime_error("recovery: cannot lock "+path); }
    }
    ~Lock() { close(fd); }
#endif
};
Recovery::~Recovery()=default;
Recovery::Recovery(const RunSettings& run,const Processes& world,Id count,unsigned dimension,
                   std::string mesh_sha256,const std::vector<OwnedNode>& nodes)
    : run_(run),world_(world),count_(count),width_(dimension+2),hash_(std::move(mesh_sha256)),
      fields_(run.directories.output),nodes_(nodes) {
    world_.local([&] {
        if (world_.rank) return;
        if (world_.size==1 && !run_.directories.recover) for (const auto& zone:run_.output_zones) {
            const auto key=output_path_key(output_filename(zone,1));
            const auto first=key.substr(0,key.find('/'));
            require(first!="recovery-0.bin" && first!="recovery-1.bin" && first!="recovery.lock",
                    "field output overlaps recovery data: "+zone.file);
        }
        create_output_directory(run_.directories.output);
        lock_.reset(new Lock(run_.directories.output+"/recovery.lock"));
        if (!run_.directories.recover)
            require(!exists(path(0)) && !exists(path(1)),"output already contains recovery data; use --recover or a new output directory");
    });
    if (hash_.empty()) hash_=archive_hash(run.mesh_path,world);
    require(hash_.size()==64,"invalid mesh fingerprint");
}
std::string Recovery::path(int slot) const {
    return run_.directories.output+"/recovery-"+std::to_string(slot)+".bin";
}
bool Recovery::due(Id accepted,double t) const {
    return run_.recovery_every_steps && (accepted%run_.recovery_every_steps==0 || t==run_.final_time);
}
void Recovery::restore(std::vector<double>& state,std::vector<double>& initial) {
    const int requested=run_.directories.recovery_slot;
    Id steps[2]={};
    double times[2]={};
    world_.local([&] {
        if (world_.rank) return;
        for (int s=0;s<2;++s) {
            if (requested>=0 && s!=requested) continue;
            if (!exists(path(s))) continue;
            try {
                std::ifstream in(path(s),std::ios::binary);
                const auto h=read_header(in,count_,width_,run_.gamma,hash_);
                steps[s]=get(&h[32]);
                times[s]=real(&h[40]);
            } catch (const std::exception& e) { std::cerr<<path(s)<<": "<<e.what()<<'\n'; }
        }
    });
    world_.broadcast(steps,2,0);
    const int first=steps[1]>steps[0]?1:0;
    for (int s:{first,1-first}) {
        if (!steps[s]) continue;
        try {
            const unsigned stride=record_size(width_);
            const auto decode=[&](const OwnedNode& n,char* row,std::size_t) {
                if (!intact(row,stride) || get(row)!=n.id)
                    throw std::runtime_error("recovery: damaged record at node "+std::to_string(n.id));
                if (get(row+8)!=n.representative)
                    throw std::runtime_error("recovery: incompatible periodic mapping at node "+std::to_string(n.id));
                if (n.id==n.representative) for (unsigned k=0;k<width_;++k) {
                    state[n.slot*width_+k]=real(row+16+8*k);
                    initial[n.slot*width_+k]=real(row+16+8*(width_+k));
                }
            };
            indexed_io::transfer(world_,path(s),indexed_io::Access::Read,stride,nodes_,{header_size},decode);
            // Verify the selected header still identifies the same completed
            // step before publishing the restored state on any rank.
            world_.local([&] {
                if (world_.rank) return;
                std::ifstream in(path(s),std::ios::binary);
                const auto h=read_header(in,count_,width_,run_.gamma,hash_);
                require(get(&h[32])==steps[s] && real(&h[40])==times[s],"checkpoint changed during read");
            });
        } catch (const std::exception& e) {
            if (!world_.rank) std::cerr<<path(s)<<": "<<e.what()<<'\n';
            continue;
        }
        world_.broadcast(&times[s],1,0);
        time=times[s]; last_=s; step=steps[s];
        require(time<=run_.final_time,"final_time precedes the saved time");
        if (!world_.rank) std::cout<<"Recovered step "<<step<<" at time "<<time<<" from "<<path(s)<<'\n';
        if (!run_.output_zones.empty()) {
            const std::string prefix=run_.directories.output+"/resumed-"+std::to_string(step)+"-";
            Id number=0;
            world_.local([&] { if (!world_.rank) number=create_resume_directory(prefix); });
            world_.broadcast(&number,1,0);
            fields_=prefix+std::to_string(number);
        }
        return;
    }
    throw std::runtime_error("recovery: no valid checkpoint in "+(requested>=0?path(requested):run_.directories.output));
}
void Recovery::save(Id accepted,double t,const std::vector<double>& state,const std::vector<double>& initial) {
    const int slot=last_==0?1:0;
    const std::string file=path(slot);
    // An uncommitted slot has no magic. Persist this before writing any new rows.
    world_.local([&] {
        if (world_.rank) return;
        std::ofstream out(file,std::ios::binary|std::ios::trunc);
        Packet blank{};
        out.write(blank.data(),header_size);
        out.seekp(offset(count_,width_)-1);
        out.put(0);
        out.close();
        require(bool(out),"cannot prepare "+file);
        sync_file(file);
    });
    const unsigned stride=record_size(width_);
    const auto encode=[&](const OwnedNode& n,char* row,std::size_t) {
        put(row,n.id); put(row+8,n.representative);
        for (unsigned k=0;k<width_;++k) {
            put_real(row+16+8*k,state[n.slot*width_+k]);
            put_real(row+16+8*(width_+k),initial[n.slot*width_+k]);
        }
        seal(row,stride);
    };
    indexed_io::transfer(world_,file,indexed_io::Access::Write,stride,nodes_,{header_size},encode);
    // MPI_File_sync has flushed the parallel write. Flush the serial stream
    // before committing its header as well.
    world_.local([&] {
        if (world_.rank) return;
        if (world_.size==1) sync_file(file);
        std::fstream out(file,std::ios::in|std::ios::out|std::ios::binary);
        const auto bytes=header(count_,width_,run_.gamma,hash_,accepted,t);
        out.write(bytes.data(),header_size);
        out.close();
        require(bool(out),"cannot commit "+file);
        sync_file(file);
    });
    last_=slot;
}
}

