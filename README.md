
<h2 align="center">使用方法</h2>

### 升级软件仓

```xml
sudo apt update
sudo apt upgrade
```

### 升级Kitware archive

```xml
wget https://apt.kitware.com/kitware-archive.sh
sudo bash kitware-archive.sh
```
### 升级相关依赖仓库

```xml
sudo apt install --no-install-recommends git cmake ninja-build gperf \
  ccache dfu-util device-tree-compiler wget \
  python3-dev python3-pip python3-setuptools python3-tk python3-wheel xz-utils file \
  make gcc gcc-multilib g++-multilib libsdl2-dev libmagic1
```

### 安装SDK工具

```xml
cd ~
wget https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.1/zephyr-sdk-0.16.1_linux-x86_64.tar.xz
wget -O - https://github.com/zephyrproject-rtos/sdk-ng/releases/download/v0.16.1/sha256.sum | shasum --check --ignore-missing
tar xvf zephyr-sdk-0.16.1_linux-x86_64.tar.xz
cd zephyr-sdk-0.16.1
./setup.sh
```

### 安装west工具
```xml
pip3 install --user -U west
echo 'export PATH=~/.local/bin:"$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 核对版本

cmake最低最低3.20.5、python3最低最低3.8、dts最低1.4.6。

```xml
cmake --version  
python3 --version  
dtc --version 
```

### 创建工作区并拉取代码
```xml
cd ~
放入代码
pip install pyelftools
```

### 初始化工作区
```xml
cd PaddleLite-on-Zephyr
cd zephyr
west init -l ~/PaddleLite-on-Zephyr/zephyr
```
执行完上面命令后，在'PaddleLite-on-Zephyr'目录下将会生成.west文件夹， 其中'config'文件中存放了west的相关配置。此时可以通过执行如下命令查看'west'配置是否成功：
```xml
west -h
```


### 验证
```xml
west build -b  qemu_cortex_a53 samples/hello_world/
west build -t  run
```
能看到正确输出Hello World! qemu_cortex_a53代表环境没问题。


### 在qemu_cortex_a53运行Paddle Lite
将git拉取的dirent.h放入到安装的sdk当中，具体放入到~/zephyr-sdk-0.16.1/aarch64-zephyr-elf/aarch64-zephyr-elf/sys-include下替换掉之前dirent.h。
```xml
rm -rf -R build
west build -b  qemu_cortex_a53 samples/paddlelite/
cd ~/PaddleLite-on-Zephyr/zephyr/build && ~/zephyr-sdk-0.16.1/sysroots/x86_64-pokysdk-linux/usr/bin/qemu-system-aarch64 -cpu cortex-a53 -nographic -machine virt,secure=on,gic-version=3 -m 4G -net none -pidfile qemu.pid -chardev stdio,id=con,mux=on -serial chardev:con -mon chardev=con,mode=readline -icount shift=4,align=off,sleep=on -rtc clock=vm -device loader,file=/path-to/PaddleLite-on-Zephyr/zephyr/samples/model/mobilenet_v1.nb,addr=0x70000000,force-raw=on -kernel ~/PaddleLite-on-Zephyr/zephyr/build/zephyr/zephyr.elf
```
~/zephyr-sdk-0.16.1代表安装sdk的位置，file=/path-to/PaddleLite-on-Zephyr/zephyr/samples/model/mobilenet_v1_opt.nb代表的是推理mobilenet_v1_opt.nb，如果想推理其他模型,模型文件放在了zephyr/samples/model。

### RK3568烧录
1.参照 https://wiki.t-firefly.com/zh_CN/ROC-RK3568-PC/debug.html 进入串口调试界面。

2.在开发板端，需要在上电时按住ctrl + c进入uboot，并使用help命令查看当前uboot是否支持tftp下载，否则需要自己修改uboot代码(当前项目已经提供uboot)。

3.默认烧录的uboot无法使用cache指令，即dcache和icache无法使用，因此需要更改uboot增加相应指令支持。修改uboot源码并且支持cache指令，然后重新编译生成uboot.img镜像(当前项目已经提供uboot)。
参考 https://wiki.t-firefly.com/zh_CN/ROC-RK3568-PC/03-upgrade_firmware.html 进行烧录。

### 在RK3568下运行paddlelite
将git拉取的dirent.h放入到安装的sdk当中，具体放入到~/zephyr-sdk-0.16.1/aarch64-zephyr-elf/aarch64-zephyr-elf/sys-include下替换掉之前dirent.h。
```xml
rm -rf -R build
west build -b  roc_rk3568_pc samples/paddlelite/
cd build
```
找到build下面的zephyr.bin文件，用tftp服务器下载生成的zephyr.bin文件，并将其加载到内存地址 0x40000000，其中根据自己tftp正确设置ip地址，配置服务器ip和当前开发板ip地址需要在同一网段。具体如下：
```xml
setenv serverip 192.168.0.102
setenv ipaddr 192.168.0.103
tftp 0x40000000 zephyr.bin;
```
tftp服务器模型文件(resnet18为例)，并将其加载到内存地址 0x70000000，具体如下：
```xml
tftp 0x70000000 resnet18.nb;
```
最终运行，具体如下：
```xml
dcache flush; icache flush; dcache off; icache off; go 0x40000000;
```














