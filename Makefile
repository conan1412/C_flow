# Tools.o : Tools.cpp
# 	g++ -c Tools.cpp -o Tools.o

# main.o : main.cpp
# 	g++ -c main.cpp -o main.o

# main.exe : main.o Tools.o
# 	g++ Tools.o main.o -o main.exe

# Tools.o : Tools.cpp

# main.o : main.cpp

# main.exe : main.o Tools.o
# 	g++ Tools.o main.o -o main.exe



# 检索./目录查找cpp后缀文件，用shell指令find
srcs := $(shell find ./ -name "*.cpp")
# srcs := $(shell find ./ -name "Tools.cpp" -or -name "main.cpp")

# 将srcs的后缀为.cpp替换为.o
objs := $(srcs:.cpp=.o)

# 将前缀替换为objs/前缀，让o文件放到objs目录下
objs := $(objs:./%=objs/%)

# 定义objs下的o文件，依赖./目录下对应的cpp文件
# $@ = 左边的生成项
# $< = 右边的依赖项第一个
objs/%.o : ./%.cpp
	@echo 编译$< 生成$@ 目录是$(dir $@)
	@mkdir -p $(dir $@)
	g++ -c $< -o $@

# $^ = 右边依赖项全部
# 我们把main放到workspace下面
workspace/main : $(objs)
	@echo 这里的依赖项所有是[$^] 目录是$(dir $@)
	@echo 链接$@
	@mkdir -p $(dir $@)
	g++ $^ -o $@

# 定义简洁指令，make main即可生成程序
main : workspace/main
	@echo 编译完成

#定义make run，编译好main后顺便执行
run : main
	@cd workspace && ./main

# 定义make clean, 清理掉编译留下的垃圾
clean :
	@rm -rf workspace/main objs/*

# 定义伪符号，这些符号不视为文件，视为指令
# 也可以说，视为永远都不存在的文件
.PHONY : main run debug clean


debug :
	@echo srcs is [$(srcs)]
	@echo objs is [$(objs)]
