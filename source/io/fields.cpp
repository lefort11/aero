#include "fields.h"
#include "../parallel/parallel_for.h"
#include <algorithm>
#include "lzma_stream.h"

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <memory>
#include <ostream>
#include <set>
#include <sstream>
#include <utility>

namespace aero {
namespace {
typedef std::uint64_t U64;
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559,"Requires IEEE-754 binary64");
[[noreturn]] void fail(const std::string& message) {throw FieldError(message);}
bool space(int c) {return c==' '||c=='\t'||c=='\r'||c=='\n'||c=='\v'||c=='\f';}
const char* location_name(FieldLocation location) {
    switch(location) {
    case FieldLocation::Node:return "NODE"; case FieldLocation::Edge:return "EDGE";
    case FieldLocation::Face:return "FACE"; case FieldLocation::Cell:return "CELL";
    }
    fail("Invalid field location");
}
U64 product(U64 a,U64 b) {
    if(b && a>std::numeric_limits<U64>::max()/b) fail("Field value count overflow");
    return a*b;
}
bool valid_utf8(const std::string& text) {
    std::size_t i=0;
    while(i<text.size()) {
        const unsigned char first=static_cast<unsigned char>(text[i++]);
        if(first<0x80) {if(first<0x20||first==0x7f)return false;continue;}
        unsigned tail;std::uint32_t code,minimum;
        if(first>=0xc2&&first<=0xdf) {tail=1;code=first&31;minimum=0x80;}
        else if(first>=0xe0&&first<=0xef) {tail=2;code=first&15;minimum=0x800;}
        else if(first>=0xf0&&first<=0xf4) {tail=3;code=first&7;minimum=0x10000;}
        else return false;
        if(tail>text.size()-i)return false;
        for(unsigned k=0;k<tail;++k) {
            const unsigned char byte=static_cast<unsigned char>(text[i++]);
            if((byte&0xc0)!=0x80)return false;
            code=(code<<6)|(byte&63);
        }
        if(code<minimum||code>0x10ffff||(code>=0xd800&&code<=0xdfff))return false;
    }
    return true;
}
bool valid_hash(const std::string& hash) {
    if(hash.size()!=64)return false;
    for(char c:hash) if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))return false;
    return true;
}
unsigned hex_value(char c) {return static_cast<unsigned>(c<='9'?c-'0':c-'a'+10);}
void check_format(FieldFormat format) {
    if(format!=FieldFormat::Text&&format!=FieldFormat::Binary&&format!=FieldFormat::Lzma)
        fail("Unsupported field encoding");
}

class Reader {
public:
    Reader(std::istream& input,const FieldReadLimits& limits)
        : input_(&input),limits_(limits),budget_(limits.max_payload_bytes),available_(limits.max_decoded_bytes),text_(false) {
        char magic[8];bytes(magic,8);
        if(std::memcmp(magic,"AEROLZMA",8)==0) {
            compressed_.reset(new io::LzmaInputBuffer(input,limits.max_decoded_bytes));
            decoded_.reset(new std::istream(compressed_.get()));decoded_->exceptions(std::ios::badbit);
            input_=decoded_.get();available_=limits.max_decoded_bytes;bytes(magic,8);
            if(std::memcmp(magic,"AEROFLDB",8)!=0)fail("LZMA fields must contain binary AEROFIELDS v1");
        }
        if(std::memcmp(magic,"AEROFIEL",8)==0) {
            char suffix[2];bytes(suffix,2);
            if(std::memcmp(suffix,"DS",2)!=0||!space(peek()))fail("Unrecognized field signature");
            text_=true;
        } else if(std::memcmp(magic,"AEROFLDB",8)!=0)fail("Unrecognized field signature");
        if(u64()!=1)fail("Unsupported field format version");
        block("TEXT");
    }
    void block(const char* label) {if(text_&&token()!=label)fail(std::string("Expected block: ")+label);}
    U64 u64() {
        if(!text_) {
            unsigned char data[8];bytes(reinterpret_cast<char*>(data),8);U64 value=0;
            for(unsigned i=0;i<8;++i)value|=U64(data[i])<<(8*i);
            return value;
        }
        const std::string value=token();U64 result=0;
        for(char c:value) {
            if(c<'0'||c>'9')fail("Invalid unsigned integer");
            const unsigned digit=static_cast<unsigned>(c-'0');
            if(result>(std::numeric_limits<U64>::max()-digit)/10)fail("Unsigned integer overflow");
            result=result*10+digit;
        }
        return result;
    }
    double real() {
        if(!text_) {const U64 bits=u64();double result;std::memcpy(&result,&bits,8);return result;}
        const std::string value=token();
        if(value=="nan")return std::numeric_limits<double>::quiet_NaN();
        if(value=="inf")return std::numeric_limits<double>::infinity();
        if(value=="-inf")return -std::numeric_limits<double>::infinity();
        std::istringstream number(value);number.imbue(std::locale::classic());double result;
        if(!(number>>result)||number.peek()!=std::char_traits<char>::eof()||!std::isfinite(result))
            fail("Invalid floating-point value");
        return result;
    }
    std::string string() {
        std::string result;
        if(!text_) {
            const U64 length=u64();
            if(length>limits_.max_string_bytes||length>result.max_size())fail("Field string exceeds limit");
            charge(length,1);result.resize(static_cast<std::size_t>(length));
            if(length)bytes(&result[0],result.size());
        } else {
            skip_space();if(get()!='"')fail("Expected quoted field string");
            for(;;) {
                int c=get();if(c==std::char_traits<char>::eof())fail("Truncated field string");
                if(c=='"')break;
                if(c=='\\') {c=get();if(c!='\\'&&c!='"')fail("Invalid field string escape");}
                if(result.size()>=limits_.max_string_bytes)fail("Field string exceeds limit");
                charge(1,1);result+=static_cast<char>(c);
            }
            if(!space(peek())&&peek()!=std::char_traits<char>::eof())fail("Missing separator after string");
        }
        if(!valid_utf8(result))fail("Invalid UTF-8 or control character in field string");
        return result;
    }
    std::string hash() {
        if(text_)return string();
        unsigned char data[32];bytes(reinterpret_cast<char*>(data),32);
        const char* digits="0123456789abcdef";std::string result;
        for(unsigned char c:data) {result+=digits[c>>4];result+=digits[c&15];}
        return result;
    }
    FieldLocation location() {
        if(!text_) {
            char value;bytes(&value,1);
            const FieldLocation result=static_cast<FieldLocation>(static_cast<unsigned char>(value));
            location_name(result);return result;
        }
        const std::string value=token();
        for(FieldLocation loc:{FieldLocation::Node,FieldLocation::Edge,FieldLocation::Face,FieldLocation::Cell})
            if(value==location_name(loc))return loc;
        fail("Invalid field location");
    }
    template<class T> void resize(std::vector<T>& array,U64 size) {
        if(size>array.max_size())fail("Field array is too large");
        charge(size,sizeof(T));array.resize(static_cast<std::size_t>(size));
    }
    void finish() {
        block("END");if(text_)skip_space();
        if(peek()!=std::char_traits<char>::eof())fail("Trailing data after fields");
        if(input_->bad())fail("Field input failed");
    }
private:
    void charge(U64 size,U64 width) {
        if(size>budget_/width)fail("Fields exceed payload limit");
        budget_-=size*width;
    }
    int peek() {return input_->peek();}
    int get() {
        if(!available_) {if(peek()==std::char_traits<char>::eof())return peek();fail("Fields exceed decoded byte limit");}
        const int c=input_->get();if(c!=std::char_traits<char>::eof())--available_;return c;
    }
    void bytes(char* data,std::size_t size) {
        if(size>available_)fail("Fields exceed decoded byte limit");
        if(size>static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))fail("Field block too large");
        if(!input_->read(data,static_cast<std::streamsize>(size)))fail("Truncated field input");
        available_-=size;
    }
    void skip_space() {while(space(peek()))get();}
    std::string token() {
        skip_space();std::string result;
        while(peek()!=std::char_traits<char>::eof()&&!space(peek())) {
            if(result.size()==128)fail("Field token too long");
            result+=static_cast<char>(get());
        }
        if(result.empty())fail("Unexpected end of field input");
        return result;
    }
    std::unique_ptr<io::LzmaInputBuffer> compressed_;
    std::unique_ptr<std::istream> decoded_;
    std::istream* input_;
    const FieldReadLimits& limits_;
    U64 budget_,available_;
    bool text_;
};

void text_real(std::ostream& stream,double value) {
    if (std::isnan(value)) stream<<"nan";
    else if (std::isinf(value)) stream<<(value<0?"-inf":"inf");
    else stream<<value;
    stream.put(' ');
}

class Writer {
public:
    Writer(std::ostream& output,bool text) : output_(output),text_(text) {
        numeric_.imbue(std::locale::classic());numeric_<<std::setprecision(std::numeric_limits<double>::max_digits10);
        literal(text_?"AEROFIELDS ":"AEROFLDB");u64(1);if(text_)literal("TEXT\n");
    }
    void literal(const char* text) {output_.write(text,static_cast<std::streamsize>(std::strlen(text)));}
    void block(const char* label) {if(text_) {output_.put('\n');literal(label);output_.put(' ');}}
    void u64(U64 value) {
        char bytes[20];
        if(text_) {
            char* end=bytes+20;char* begin=end;
            do {*--begin=static_cast<char>('0'+value%10);value/=10;}while(value);
            output_.write(begin,end-begin);output_.put(' ');
        } else {
            for(unsigned i=0;i<8;++i)bytes[i]=static_cast<char>((value>>(8*i))&255);
            output_.write(bytes,8);
        }
    }
    void real(double value) {
        if(!text_) {U64 bits;std::memcpy(&bits,&value,8);u64(bits);return;}
        numeric_.clear();numeric_.str("");text_real(numeric_,value);
        const auto s=numeric_.str();output_.write(s.data(),static_cast<std::streamsize>(s.size()));
    }
    void values(const std::vector<double>& data,U64 count) {
        if (!text_) {
            for (double value:data) real(value);
            return;
        }
        // Format disjoint blocks in parallel, then write them in file order.
        const std::size_t chunk=512,batch=32*chunk;
        for (std::size_t start=0;start<data.size();) {
            const auto size=std::min(batch,data.size()-start);
            std::vector<std::string> parts((size+chunk-1)/chunk);
            parallel_for(parts.size(),[&](std::size_t part) {
                std::ostringstream s;
                s.imbue(std::locale::classic());
                s<<std::setprecision(std::numeric_limits<double>::max_digits10);
                const auto end=start+std::min((part+1)*chunk,size);
                for (std::size_t i=start+part*chunk;i<end;++i) {
                    text_real(s,data[i]);
                    if ((i+1)%8==0 || (i+1)%count==0) s.put('\n');
                }
                parts[part]=s.str();
            },8);
            for (const auto& part:parts) output_.write(part.data(),static_cast<std::streamsize>(part.size()));
            start+=size;
        }
    }
    void string(const std::string& value) {
        if(text_) {
            output_.put('"');for(char c:value) {if(c=='"'||c=='\\')output_.put('\\');output_.put(c);}literal("\" ");
        } else {
            if(value.size()>static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()))fail("Field string too large");
            u64(value.size());output_.write(value.data(),static_cast<std::streamsize>(value.size()));
        }
    }
    void hash(const std::string& value) {
        if(text_)string(value);
        else for(std::size_t i=0;i<64;i+=2)output_.put(static_cast<char>((hex_value(value[i])<<4)|hex_value(value[i+1])));
    }
    void location(FieldLocation value) {
        if(text_) {literal(location_name(value));output_.put(' ');}
        else output_.put(static_cast<char>(value));
    }
    void finish() {block("END");if(text_)output_.put('\n');if(!output_)fail("Field output failed");}
private:
    std::ostream& output_;
    bool text_;
    std::ostringstream numeric_;
};
void write_payload(const FieldSnapshot& snapshot,std::ostream& out,bool text) {
    Writer writer(out,text);
    writer.block("DIMENSION");writer.u64(snapshot.dimension);
    writer.block("MESH_SHA256");writer.hash(snapshot.mesh_sha256);
    writer.block("COUNTS");for(U64 value:snapshot.counts)writer.u64(value);
    writer.block("STEP");writer.u64(snapshot.step);
    writer.block("TIME");writer.real(snapshot.time);
    writer.block("FIELDS");writer.u64(snapshot.fields.size());
    for(const Field& field:snapshot.fields) {
        writer.block("FIELD");writer.string(field.name);
        writer.block("LOCATION");writer.location(field.location);
        writer.block("COMPONENTS");writer.u64(field.components.size());for(const std::string& c:field.components)writer.string(c);
        writer.block("UNITS");writer.string(field.units);
        writer.block("VALUES");writer.u64(snapshot.entity_count(field.location));
        writer.block("DATA");
        const U64 count=snapshot.entity_count(field.location);
        writer.values(field.values,count);
        writer.block("END_FIELD");
    }
    writer.finish();
}
void write_encoded(const FieldSnapshot& snapshot,std::ostream& out,FieldFormat format) {
    if(format==FieldFormat::Lzma) {
        io::LzmaOutputBuffer buffer(out);std::ostream compressed(&buffer);
        compressed.exceptions(std::ios::badbit|std::ios::failbit);
        write_payload(snapshot,compressed,false);buffer.finish();
    } else write_payload(snapshot,out,format==FieldFormat::Text);
}
} // namespace

U64 FieldSnapshot::entity_count(FieldLocation location) const {
    location_name(location);return counts[static_cast<std::size_t>(location)];
}
void FieldSnapshot::validate() const {
    if(dimension<1||dimension>3)fail("Field dimension must be 1, 2 or 3");
    if(!valid_hash(mesh_sha256))fail("Invalid mesh SHA-256; expected 64 lowercase hex characters");
    if(!std::isfinite(time))fail("Field snapshot time must be finite");
    std::set<std::pair<FieldLocation,std::string> > names;
    for(const Field& field:fields) {
        const U64 count=entity_count(field.location);
        if(field.name.empty()||!valid_utf8(field.name)||!valid_utf8(field.units))fail("Invalid field name or units");
        if(!names.insert(std::make_pair(field.location,field.name)).second)fail("Duplicate field name at location");
        if(field.components.empty())fail("Field must have components");
        std::set<std::string> components;
        for(const std::string& c:field.components) {
            if(c.empty()||!valid_utf8(c))fail("Invalid component name");
            if(!components.insert(c).second)fail("Duplicate component name");
        }
        if(product(count,field.components.size())!=field.values.size())fail("Field values size mismatch");
    }
}
FieldSnapshot FieldSnapshot::read(std::istream& input,const FieldReadLimits& limits) try {
    Reader reader(input,limits);FieldSnapshot result;
    reader.block("DIMENSION");const U64 dimension=reader.u64();
    if(dimension<1||dimension>3)fail("Field dimension must be 1, 2 or 3");
    result.dimension=static_cast<unsigned>(dimension);
    reader.block("MESH_SHA256");result.mesh_sha256=reader.hash();
    if(!valid_hash(result.mesh_sha256))fail("Invalid mesh SHA-256");
    reader.block("COUNTS");for(U64& count:result.counts)count=reader.u64();
    reader.block("STEP");result.step=reader.u64();reader.block("TIME");result.time=reader.real();
    if(!std::isfinite(result.time))fail("Field snapshot time must be finite");
    reader.block("FIELDS");const U64 field_count=reader.u64();
    if(field_count>limits.max_fields)fail("Field count exceeds limit");
    reader.resize(result.fields,field_count);
    for(Field& field:result.fields) {
        reader.block("FIELD");field.name=reader.string();
        reader.block("LOCATION");field.location=reader.location();
        reader.block("COMPONENTS");const U64 components=reader.u64();
        if(!components||components>limits.max_components)fail("Field component count invalid or exceeds limit");
        reader.resize(field.components,components);for(std::string& c:field.components)c=reader.string();
        reader.block("UNITS");field.units=reader.string();
        reader.block("VALUES");const U64 count=reader.u64();
        if(count!=result.entity_count(field.location))fail("Field entity count mismatch");
        reader.resize(field.values,product(count,components));
        reader.block("DATA");for(double& value:field.values)value=reader.real();reader.block("END_FIELD");
    }
    reader.finish();result.validate();return result;
} catch(const io::CompressionError& error) {throw FieldError(error.what());}
  catch(const std::ios_base::failure& error) {throw FieldError(std::string("Field input failed: ")+error.what());}
FieldSnapshot FieldSnapshot::read(const std::string& path,const FieldReadLimits& limits) {
    if(path.find('\0')!=std::string::npos)fail("NUL in field path");
    std::ifstream input(path.c_str(),std::ios::binary);if(!input)fail("Cannot open fields for reading: "+path);
    return read(input,limits);
}
void FieldSnapshot::write(std::ostream& output,FieldFormat format) const try {
    validate();check_format(format);write_encoded(*this,output,format);
} catch(const io::CompressionError& error) {throw FieldError(error.what());}
  catch(const std::ios_base::failure& error) {throw FieldError(std::string("Field output failed: ")+error.what());}
void FieldSnapshot::write(const std::string& path,FieldFormat format) const {
    validate();check_format(format);
    if(path.find('\0')!=std::string::npos)fail("NUL in field path");
    std::ofstream output(path.c_str(),std::ios::binary|std::ios::trunc);
    if(!output)fail("Cannot open fields for writing: "+path);
    write(output,format);output.close();if(!output)fail("Cannot finish writing fields: "+path);
}
} // namespace aero
