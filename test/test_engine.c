#include "test_engine.h"
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include "../src/engine.h"


void test_add(void)
{
    Value *a = value_new(-4.0, NULL, 0, "");
    Value *b = value_new(2.0, NULL, 0, "");
    Value *c = value_add(a, b);
    backward(c);

    float tol = 1e-4;
    assert(fabs(c->data - (-2.0)) < tol);
    printf("c->data: %.6f, expected: -2.000000\n", c->data);
    assert(fabs(a->grad - 1) < tol);
    printf("a->grad: %.6f, expected: 1.000000\n", a->grad);
    assert(fabs(b->grad - 1) < tol);
    printf("b->grad: %.6f, expected: 1.000000\n", b->grad);

    printf("test_add passed\n");
}
