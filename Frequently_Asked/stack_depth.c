/*Tell me the logic how can you find depth of a stack?
Example In a program their is foo1() function calling foo2() function  calling foo3() in any function 
write a logic to know the depth of stack.*/

global stack_base_ptr

main() {
    int main_var;
    stack_base_ptr = &main_var;  // stack base

    foo1();
}

foo1() {
    int var1;
    print("Depth at foo1 =", abs(&var1 - stack_base_ptr));
    foo2();
}

foo2() {
    int var2;
    print("Depth at foo2 =", abs(&var2 - stack_base_ptr));
    foo3();
}

foo3() {
    int var3;
    print("Depth at foo3 =", abs(&var3 - stack_base_ptr));
}
