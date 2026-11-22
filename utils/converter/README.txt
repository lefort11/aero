Конвертер Gmsh / CGNS / another -> AEROMESH v3
===================================

Утилита работает на CPU, использует C++11 и не требует CUDA.
CGNS/HDF5 подключаются только к конвертеру, не к классу сетки.

Запуск из корня проекта:
  bin/mesh_converter.exe utils/converter/examples/line.msh build/line.mesh --format text
  bin/mesh_converter.exe input.msh output.mesh
  bin/mesh_converter.exe input.cgns output.mesh --format binary
  bin/mesh_converter.exe input.msh output.mesh --format lzma
  bin/mesh_converter.exe input.cgns output.mesh --base 1 --zone 2
  bin/mesh_converter.exe path/to/me1_ntte output.mesh
  bin/mesh_converter.exe path/to/me1_ntte/mesh.txt output.mesh --format text
  bin/mesh_converter.exe path/to/MESH output-bin.mesh --from another --format binary
  bin/mesh_converter.exe --help

Первые два аргумента обязательны: INPUT (входная сетка), OUTPUT (выходной путь).
Без аргументов или при отсутствии одного из них печатается краткая справка
в stderr, код возврата 1. По --help или -h справка идёт в stdout, код 0.

Формат входа определяется по .msh/.cgns, каталогу с mesh.txt или пути mesh.txt;
либо задаётся через --from gmsh|cgns|another.
Выход: text, binary или lzma (сжатая бинарная сетка v3), по умолчанию binary.
LZMA подключена через общую библиотеку aero_io; описание: source/io/README.txt.
Существующий выходной файл не перезаписывается. Вход полностью читается и
проверяется до открытия выходного файла. Ошибка возвращает код 1.
Ошибка записи может оставить неполный выходной файл.

Границы и масштаб:
- Gmsh: имена физических групп размерности cell_dimension - 1.
  Если имя группы отсутствует, используется physical_<номер>.
- CGNS: имена узлов BC_t; типы граничных условий не переносятся.
  Для Vertex список узлов отмечает все внешние грани, целиком лежащие
  в этом списке. Пересекающиеся назначения разных имён отклоняются.
- Внешние грани без меток получают имя unmarked. Меняется через
  --unmarked NAME. В библиотечном API пустой Options::unmarked_name
  запрещает немаркированные внешние грани.
- В файл записываются координаты узлов исходной сетки.
- Одна грань хранится один раз, её ориентация задаётся относительно элемента.
  Вход должен иметь согласованную ориентацию элементов. Конвертер не
  исправляет инвертированные/вырожденные элементы и не проверяет пересечения.
- Метки объёмных областей и данные решения в формат сетки не переносятся.

Поддерживаемые входы:
another:
  Обычная 2D/3D сетка: mesh.txt, coordinate.msh, topo.msh, bctopo.msh,
  label.txt. В mesh.txt указывается MeshFormat text или MeshFormat 0.
  Координаты и обе таблицы связности читаются из текстовых .msh. Кодировка имён — UTF-8.
  Идентификаторы узлов начинаются с нуля. Координаты записаны по узлам:
  x y или x y z. NumCoords задаёт размерность.
  Размеры: Num_Node / Num_Elem / Num_BndFace либо Nn / Nt / NFaceBC.
  MeshType=HYBRID: каждая запись топологии начинается с количества узлов.
  2D: треугольники, четырёхугольники, полигоны; 3D: тетраэдры (4),
  пирамиды (5), призмы (6), гексаэдры (8).
  Без MeshType: однородные квады в 2D, гексы в 3D; префикса размера нет.
  MeshType=quad/hex/hexa также выбирает однородный вариант.
  У квадов и четырёхугольных оснований объёмных элементов порядок узлов
  тензорный: 0,1,2,3 переводятся в циклический 0,1,3,2; верх гекса аналогично.
  Треугольные основания призм и тетраэдры используют обычную симплексную
  нумерацию. В bctopo вершины уже перечислены по контуру, затем идёт ID метки.
  3D произвольные многогранники из этого формата пока не читаются: одного
  списка узлов недостаточно для восстановления их граней.
  label.txt: необязательное число записей, затем ID NAME [aliases...] #comment.
  Используется первое имя; дополнительные слова не становятся типами ГУ.
  Внешние грани должны быть перечислены полностью, без повторов; каждому
  используемому ID нужно имя. Неполная граница вызывает ошибку.
  Варианты *_per, буферные зоны и периодические соответствия пока отклоняются.

Gmsh:
  MSH 2.2 и 4.1, ASCII и binary, оба порядка байтов, data-size=8.
  Разреженные номера узлов и элементов, PhysicalNames, Entities,
  параметрические координаты узлов (дополнительные параметры пропускаются).
  Линейные точки, отрезки, треугольники, четырёхугольники, тетраэдры,
  гексаэдры, треугольные призмы, пирамиды.
  Размерность определяется по максимальной размерности элементов.
  Несколько физических групп у одной граничной сущности отклоняются.
  Высокие порядки, версии MSH 4.0/4.2, периодические и разбитые на разделы
  сетки пока отклоняются. Неизвестные бинарные секции отклоняются;
  дополнительные ASCII-секции пропускаются, кроме связности разделов/периодики.

CGNS (через библиотеку CGNS):
  ADF и HDF5, одна выбранная база и зона.
  Если баз/зон несколько, необходим явный --base/--zone; объединения зон нет.
  Структурированные 1D/2D/3D сетки преобразуются в отрезки/квады/гексы.
  Неструктурированные линейные NODE/BAR_2/TRI_3/QUAD_4/TETRA_4/HEXA_8/
  PENTA_6/PYRA_5, секции MIXED, полигоны NGON_n в 2D,
  многогранники NGON_n + NFACE_n в 3D со знаковыми ссылками на грани.
  BC PointList/PointRange и прежние ElementList/ElementRange.
  Для неструктурированных сеток: Vertex и ссылки на граничные элементы
  FaceCenter/EdgeCenter. Для структурированных BC: только Vertex.
  Декартовы CoordinateX/Y/Z; недостающие неиспользуемые координаты = 0.
  Межзонная/периодическая связность, высокие порядки и NGON-объёмы
  без NFACE пока отклоняются. Имена секций Elements_t не считаются BC.
  Читается исходная сетка GridCoordinates, без временных состояний решения.

Сборка GNU Make из корня:
  make converter
  make converter-test
  make converter-debug-test
  make test
  make debug-test
Также можно запускать make / make test из utils/converter.

В Linux make converter и make programs по умолчанию собирают статические
CGNS 4.4.0 и HDF5 1.14.6 из архивов в third_party, рядом с lzma.
Нужны GNU Make, компиляторы C и C++, CMake и sha256sum. Библиотеки
устанавливаются в build/deps/linux/install. В Windows GNU Make включает
CGNS при наличии build/deps/install/include/cgnslib.h.
Для своей установки:
  make converter WITH_CGNS=1 CGNS_PREFIX=/path/to/prefix
  make converter WITH_CGNS=1 CGNS_CPPFLAGS="-I/path/include" CGNS_LDLIBS="-L/path/lib -lcgns -lhdf5 ..."
Зависимые системные библиотеки задаются согласно вашей сборке CGNS/HDF5.
Без CGNS:
  make converter WITH_CGNS=0
Gmsh продолжает работать; попытка чтения CGNS сообщает, как включить поддержку.

Сборка зависимостей на Windows (всё остаётся в build/deps):
  powershell -File utils/converter/build_dependencies.ps1 -Toolchain MinGW
  powershell -File utils/converter/build_dependencies.ps1 -Toolchain MSVC
При необходимости передать -CMake, -CCompiler, -Make или -Generator.
Для MinGW нужны gcc и mingw32-make в PATH, включая DLL среды компилятора.
Windows и Linux используют third_party/CGNS-4.4.0.zip и
third_party/HDF5-1.14.6.zip. Перед распаковкой проверяется SHA-256.
CMake применяется только для сторонних библиотек; проект собирается Makefile/MSVS.
Локальная минимальная сборка HDF5 не включает zlib/szip: сжатые этими фильтрами
массивы потребуют другую сборку HDF5 с соответствующей поддержкой.

Visual Studio:
  Открыть msvs/aero.sln; проекты mesh_converter и converter_tests.
  Конвертер и тесты публикуются в bin.
  Тесты запускаются после сборки converter_tests.
  CGNS автоматически ищется в build/deps/install-msvc-mt/<Configuration>.
  Скрипт build_dependencies.ps1 -Toolchain MSVC собирает CGNS и HDF5
  со статическим CRT (/MT для Release, /MTd для Debug).
  Для своей установки передать /p:WithCGNS=true /p:CGNSRoot=...
  При других именах библиотек: /p:HDF5Library=... или /p:CGNSLibraries=...
  /p:WithCGNS=false отключает CGNS.
  Заголовки/библиотеки должны соответствовать компилятору, архитектуре и
  конфигурации. MinGW-библиотеки не используются в MSVC.

Разделение кода:
  converter.h      публичные функции чтения и параметры
  topology.cpp     общая сборка топологии и меток
  gmsh.cpp         чтение MSH
  cgns.cpp         адаптер CGNS, отключаемый при сборке
  another.cpp     чтение текстовых сеток another
  main.cpp         точка входа
  cli.h, cli.cpp   разбор командной строки и справка
  converter_tests.cpp  модульные и интеграционные тесты

Спецификации:
  https://gmsh.info/doc/texinfo/gmsh.html#MSH-file-format
  https://cgns.org/cgns-archives/CGNS_docs_current/midlevel/grid.html
  https://cgns.org/cgns-archives/CGNS_docs_current/midlevel/bc.html
Сторонние библиотеки:
  https://github.com/CGNS/CGNS/releases/tag/v4.4.0
  https://github.com/HDFGroup/hdf5/releases/tag/hdf5_1.14.6
Лицензии сохранены в исходных архивах third_party; при распространении
библиотек/исполняемых файлов необходимо прилагать требуемые ими уведомления.
