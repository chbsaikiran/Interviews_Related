what is difference between complier linker and loader in C language, give example where the code is spread in to more than one folder and multiple files are in foldrs.

Sure! Let’s carefully differentiate **compiler, linker, and loader** in C, and then see an example with multiple files and folders.

---

### **1. Compiler**

* Translates **C source code (`.c`)** into **object code (`.o`)**.
* Performs **syntax checking, type checking, optimization**, and generates **machine instructions**.
* Each `.c` file is compiled **independently**.

**Example:**

```bash
gcc -c src/main.c -o obj/main.o
gcc -c src/utils/math.c -o obj/math.o
```

* Here `-c` tells GCC to **compile only**, producing `.o` object files.

---

### **2. Linker**

* Combines **multiple object files** into a **single executable**.
* Resolves **symbols** (functions, global variables) across files.
* Handles **libraries** (`.a` static, `.so` dynamic).

**Example:**

```bash
gcc obj/main.o obj/math.o -o bin/myprogram
```

* The linker makes sure calls like `add()` in `main.c` are linked to `add()` in `math.c`.

* **Without linking**, the program will not run if functions/variables are used across files.

---

### **3. Loader**

* Part of the **operating system**.

* Loads the executable into **memory** for execution.

* Sets up **stack, heap, global variables**, and starts program execution at `main()`.

* This happens **after compilation and linking**, during program launch.

---

### **4. Example with multiple folders**

Project structure:

```
project/
│
├─ src/
│   ├─ main.c
│   ├─ utils/
│   │   └─ math.c
│
├─ include/
│   └─ math.h
│
├─ obj/
│   └─ (object files here)
│
└─ bin/
    └─ myprogram (executable)
```

**Files:**

**include/math.h**

```c
#ifndef MATH_H
#define MATH_H

int add(int a, int b);
int subtract(int a, int b);

#endif
```

**src/utils/math.c**

```c
#include "math.h"

int add(int a, int b) {
    return a + b;
}

int subtract(int a, int b) {
    return a - b;
}
```

**src/main.c**

```c
#include <stdio.h>
#include "math.h"

int main() {
    int x = 5, y = 3;
    printf("Sum = %d\n", add(x, y));
    printf("Difference = %d\n", subtract(x, y));
    return 0;
}
```

---

**Build commands:**

```bash
# Step 1: Compile each source file into object files
gcc -Iinclude -c src/main.c -o obj/main.o
gcc -Iinclude -c src/utils/math.c -o obj/math.o

# Step 2: Link object files into executable
gcc obj/main.o obj/math.o -o bin/myprogram

# Step 3: Run (loader loads it into memory)
./bin/myprogram
```

**Explanation:**

1. **Compiler** → `main.c` → `main.o`, `math.c` → `math.o`
2. **Linker** → `main.o + math.o` → `myprogram` executable
3. **Loader** → `myprogram` loaded into memory, ready to run

---

### **Summary Table**

| Component | Task                                                      | Input      | Output          |
| --------- | --------------------------------------------------------- | ---------- | --------------- |
| Compiler  | Converts C code to object code                            | `.c` files | `.o` files      |
| Linker    | Combines `.o` files, resolves symbols                     | `.o` files | Executable      |
| Loader    | Loads executable into memory, sets up runtime environment | Executable | Running program |

---

If you want, I can also make a **diagram showing multiple folders, compilation, linking, and loader steps**, which makes this extremely visual and easy to understand.




Typical process memory layout looks like:

```
+-------------------+
| Text segment      |  (program code, instructions)
+-------------------+
| Read-only data    |  (string literals, const globals/statics)
+-------------------+
| Initialized data  |  (global/static variables with initial values)
+-------------------+
| Uninitialized data|  (BSS: global/static variables = 0 or uninit)
+-------------------+
| Heap              |  (malloc/new allocations, grows upward)
+-------------------+
| Stack             |  (local variables, grows downward)
