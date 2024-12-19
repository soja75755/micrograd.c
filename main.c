#include <stdio.h>
#include "src/engine.h"
#include "test/test_engine.h"

int main(void)
{
    test_add();
    test_mul();
    test_pow();
    test_relu();

    test_neg();
    test_sub();
    test_div();

    printf("All tests passed!\n");
    return 0;
}
