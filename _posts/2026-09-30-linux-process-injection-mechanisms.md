---
layout: "post"
title: "Linux 进程交互与代码注入机制笔记"
date: "2026-09-30"
description: "整理 ptrace、procfs 与 process_vm_writev，以及进程内存和执行流的交互机制。"
categories: ["Architecture"]
tags: ["linux", "kernel", "ptrace", "process"]
permalink: "/blog/linux-process-injection-mechanisms/"
lang: "zh-CN"
notes_import: true
source_path: "Arch/Linux/injection.md"
toc:
  beginning: true
---

injection是一个进程向另一个正在运行的进程注入hook来改变其行为。

Linux上injection的相对Windows上来说要更困难一些。Windows上可以给remote process分配内存，修改内存；但是Linux没有给remote process分配内存的API。因此，在Windows上injection的过程是：allocate → write → execute；而在Linux上，没有allocate的手段，只能：overwrite → execute → recover。

### 1. Remote process interaction methods

在Linux上，与remote process的内存交互的主要手段只有三个：_ptrace_, _procfs_, and _process_vm_writev_。

#### 1.1 ptrace

ptrace是一个syscall，原型是：

```c
#include <sys/ptrace.h>

long ptrace(enum __ptrace_request op, pid_t pid,
                   void *addr, void *data);
```

用途是Attach to the remote process。

op有多种：

1. PTRACE_PEEKTEXT/PTRACE_PEEKDATA

从指定的addr处（位于remote process的地址空间内）读一个word。由于Linux的进程地址空间中不区分代码段和数据段（段基址是一样的），所以这两个op是等效的。

2. PTRACE_GETREGS/PTRACE_GETFPREGS

从被trace的remote process读一个结构体，里面包含了remote process的通用寄存器/浮点寄存器：

```c
// Attach to the remote process
ptrace(PTRACE_ATTACH, pid, NULL, NULL);
wait(NULL);

// Get registers state
struct user_regs_struct regs;
ptrace(PTRACE_GETREGS, pid, NULL, &regs);
```

3.  PTRACE_SETREGS/PTRACE_SETFPREGS

修改remote process的通用寄存器/浮点寄存器。

4. PTRACE_ATTACH

attach到指定pid的remote process上。

5. PTRACE_DETACH

从attach的remote process上detach。

6. PTRACE_INTERRUPT

stop当前attach的remote process。如果这个tracee在执行syscall，被打断的syscall会在tracee重启后被重新调用。（本来进入syscall之前，rip会+2，因为syscall指令2字节，但是如果中途被打断，会重新执行这个syscall确保完成，因此rip-2）。

#### 1.2 procfs

procfs是一个特殊的伪文件系统，是一个所有正在运行的进程的接口。这个文件系统挂载在/proc目录下。

/proc目录下要么是pid的目录，要么是某个程序名的目录。

![image-20250707100003993]({{ '/assets/blog/linux-process-injection-mechanisms/image-20250707100003993.png' | relative_url }}){: .img-fluid loading="lazy" }

每个子目录下都有一些文件记录进程信息：

![image-20250707100349989]({{ '/assets/blog/linux-process-injection-mechanisms/image-20250707100349989.png' | relative_url }}){: .img-fluid loading="lazy" }

这里主要关注mem和maps文件。

mem文件存储了remote process的整个内存内容。用特定的偏移访问mem文件等效于用这个偏移访问remote process的虚拟地址空间：

![image-20250707100855292]({{ '/assets/blog/linux-process-injection-mechanisms/image-20250707100855292.png' | relative_url }}){: .img-fluid loading="lazy" }

maps文件则是显示了remote process的整个内存布局：

![image-20250707101101604]({{ '/assets/blog/linux-process-injection-mechanisms/image-20250707101101604.png' | relative_url }}){: .img-fluid loading="lazy" }

#### 1.3 process_vm_writev

process_vm_writev也是一个syscall，允许向remote process的地址空间中写数据。

原型是：

```c
#include <sys/uio.h>

ssize_t process_vm_readv(pid_t pid,
                      const struct iovec *local_iov,
                      unsigned long liovcnt,
                      const struct iovec *remote_iov,
                      unsigned long riovcnt,
                      unsigned long flags);
ssize_t process_vm_writev(pid_t pid,
                      const struct iovec *local_iov,
                      unsigned long liovcnt,
                      const struct iovec *remote_iov,
                      unsigned long riovcnt,
                      unsigned long flags);
```

使用的实例：

```c
// Initialize local and remote iovec structs used to perform the syscall
  struct iovec local[1];
  struct iovec remote[1];

  // Place our data in the local iovec
  local[0].iov_base = data;
  local[0].iov_len = data_len;

  // Point the remote iovec to the address in the remote process
  remote[0].iov_base = (void *)remote_address;
  remote[0].iov_len = data_len;

  // Write the local data to the remote address
  process_vm_writev(pid, local, 1, remote, 1, 0);
```

### 2. Writing code to a remote process

Linux上code injection第一步就是向remote process的内存中写数据。

已经说过，Linux没有提供在remote process中allocate新内存的API，因此只能使用已有的内存区域来写数据。

由于写入的是注入代码，应该需要找到一个有X权限的区域，这一步可以通过解析procfs下maps文件做到。

#### 2.1 Writing code to RX memory

可用ptrace和procfs做到。

一般来说，不会有WX的区域。如果一个内存区域是X的，想要执行这里的代码，还需要有R权限来保证读取，否则其实也是不能执行的。

虽然内存区域是RX，没有W权限，ptrace和procfs的mem也是可以绕过权限直接写数据的。

用ptrace的方法就是设置op为POKE类型即可：

```c
ptrace(PTRACE_ATTACH, pid, NULL, NULL);
  wait(NULL);

  // write payload to remote address
  for (size_t i = 0; i < payload_size; i += 8, payload++)
  {
    ptrace(PTRACE_POKETEXT, pid, address + i, *payload);
  }
```

用procfs mem的方法就是用正常写文件的方法去写mem文件：

```c
// Open the process mem file
FILE *file = fopen("/proc/<pid>/mem", "w");

// Set the file index to our required offset, representing the memory address
fseek(file, address, SEEK_SET);

// Write our payload to the mem file
fwrite(payload, sizeof(char), payload_size, file);
```

#### 2.2 Write code to WX memory

这里所谓的WX应该还是有R权限的（查看了具体的maps文件，查看的maps文件中内存的所有区域都是可读的）。

如果有W权限，那么ptrace, procfs mem, process_vm_writev都是可以用的。

过程都差不多，就是先利用maps文件去找符合要求的区域，然后写数据。

### 3. Hijacking remote execution flow

完成写操作之后，就需要劫持remote process的执行流了。

#### 3.1 Modifying the process instruction pointer

其实就是修改RIP寄存器，偷换下一条执行的指令。这种方式只能通过ptrace做到：

```c
// Get old register state.
struct user_regs_struct regs;
ptrace(PTRACE_GETREGS, pid, NULL, &regs);

// Modify the instruction pointer to point to our payload
regs.rip = payload_address + 2;

// Modify the registers
ptrace(PTRACE_SETREGS, pid, NULL, &regs);
```

除了SETREGS，ptrace还可以通过POKEUSER做到：

```c
// calculate the offset of the RIP register, based on the USER struct definition
rip_offset = 16 * sizeof(unsigned long);
ptrace(PTRACE_POKEUSER, pid, rip_offset, payload_address + 2);
```

> **Note**
>
> 为什么要把RIP设置为payload_address + 2？
>
> 先解释一下最安全的payload布局：开头两条nop指令（共占2字节），然后开始真正的payload指令。
>
> 原因在于如果interrupt我们attach的remote process时，它正在syscall执行过程中，那么中断处理结束后是会把RIP-2以重新执行syscall确保完成的。
>
> 因此，如果我们在这种情况下把RIP设置为payload的起始地址，中断结束后RIP就变成了payload-2，是一个未知的地址，很可能引发段错误。所以在payload开头加两条nop，可以解决这个问题：正常情况下就直接从真正的payload处开始执行；RIP-2的情况下就先执行两条安全的nop指令再开始真正的payload。

#### 3.2 Modify current instruction

简单来说，就是通过查看procfs下的syscall文件：

Another interesting file in procfs is the syscall file. This file holds information about the syscall that is currently executed by the process — the syscall number, the arguments that were passed to it, the stack pointer, and (most interesting for our cause) the process instruction pointer (Figure 10). [查看原文图10](https://www.akamai.com/blog/security-research/2024/nov/the-definitive-guide-to-linux-process-injection)

这里有rip的地址（最后一个），这个方法不修改RIP，而是写RIP指示的位置，可以通过写mem文件完成：

```c
// Suspend the process by sending a SIGSTOP signal
kill(pid, SIGSTOP);

// Open the syscall file
FILE *syscall_file = fopen("/proc/<pid>/syscall", "r");

// Extract the instruction pointer from the syscall file
long instruction_pointer = ...

// Write our payload to the address of the current instruction pointer using
procfs mem
FILE *mem_file = fopen("/proc/<pid>/mem", "w");
fseek(mem_file, instruction_pointer, SEEK_SET);
fwrite(payload, sizeof(char), payload_size, mem_file);

// Resume execution by sending a SIGCONT signal
kill(pid, SIGCONT);
```

#### 3.3 Stack hijacking

stack的区域可以通过maps文件指示的布局来确认，虽然stack是没有X权限的，但是可以修改压栈的返回地址来跳转到payload地址：

When the function finishes execution, the processor takes this return address from the stack and jumps to it (Figure 12). [查看原文图12](https://www.akamai.com/blog/security-research/2024/nov/the-definitive-guide-to-linux-process-injection)

To abuse this mechanism, we can identify a return address on the stack and overwrite it with a new address that points to our shellcode. As soon as the current function finishes execution, our code will run (Figure 13). [查看原文图13](https://www.akamai.com/blog/security-research/2024/nov/the-definitive-guide-to-linux-process-injection)

具体过程为：

1. 先发送kill(pid,SIGSTOP)暂停remote process
2. 通过解析procfs下的maps文件确认栈指针
3. 扫描栈，确认返回地址
4. 写对应内存
5. 修改返回地址
6. kill(pid,SIGCONT)以唤醒remote process

#### 3.4 ROP

概览 ROP 链的执行流程：

1. 程序返回到栈顶的 `pop rdi; ret` gadget 地址。
2. `pop rdi` 将 Shellcode 的地址载入 `rdi`。
3. `ret` 到栈顶的 `pop rsi; ret` gadget 地址。
4. `pop rsi` 将长度载入 `rsi`。
5. `ret` 到栈顶的 `pop rdx; ret` gadget 地址。
6. `pop rdx` 将保护权限载入 `rdx`。
7. `ret` 到栈顶的 `mprotect` 函数的真实地址。
8. `mprotect` 函数被调用，它会根据 `rdi`, `rsi`, `rdx` 的值修改对应内存区域的权限。
9. `mprotect` 函数执行完毕后返回。它会将控制流返回到 `mprotect` 函数被 ROP 链调用时的栈上下一地址。
10. **跳转到 Shellcode：** 此时，ROP 链的下一步通常是在 `mprotect` 函数地址之后放置 Shellcode 的起始地址。`mprotect` 返回后，程序会直接跳转到这个地址，执行 Shellcode。

示例 ROP 链（概念性栈布局）：

```
[ ... 填充直到栈溢出点 ... ]

[ 地址 of pop rdi; ret ]  <-- 第一个 gadget
[ arg1 (addr: Shellcode_Addr) ]

[ 地址 of pop rsi; ret ]  <-- 第二个 gadget
[ arg2 (len: 0x1000) ]

[ 地址 of pop rdx; ret ]  <-- 第三个 gadget
[ arg3 (prot: 0x7) ]

[ 地址 of mprotect() ]    <-- 调用 mprotect
[ 地址 of Shellcode_Addr ] <-- mprotect 返回后跳转到这里执行 shellcode
[ ... Shellcode ... ]
```

#### 3.5 GOT hijacking

Whenever the program calls a function from a remote library, it resolves its memory address by accessing the GOT (Figure 15). [查看原文图15](https://www.akamai.com/blog/security-research/2024/nov/the-definitive-guide-to-linux-process-injection)

GOT表通常是可写的，可以修改GOT表中函数调用的地址来劫持：

The GOT memory is normally writable, meaning that an attacker can overwrite any of the addresses inside it with the address of their payload. The next time the function is called by the process, the attacker code will execute instead (Figure 16). [查看原文图16](https://www.akamai.com/blog/security-research/2024/nov/the-definitive-guide-to-linux-process-injection)

过程：

1. kill(pid,SIGSTOP)
2. 通过maps文件来确认GOT所在的内存位置
3. 修改GOT中的地址为payload的地址
4. kill(pid,SIGCONT)

如果remote process对应的程序编译时启用了RELRO（relocation readonly），那么GOT不可修改，但是调用的别的库如果没启用RELRO，还是可以修改这些库的GOT完成劫持。
