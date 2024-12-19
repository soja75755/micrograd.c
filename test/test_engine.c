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
    assert(fabs(a->grad - (1.0)) < tol);
    printf("a->grad: %.6f, expected: 1.000000\n", a->grad);
    assert(fabs(b->grad - (1.0)) < tol);
    printf("b->grad: %.6f, expected: 1.000000\n", b->grad);

    printf("test_add passed\n");
}

void test_mul(void)
{
    Value *a = value_new(-4.0, NULL, 0, "");
    Value *b = value_new(2.0, NULL, 0, "");
    Value *c = value_mul(a, b);
    backward(c);

    float tol = 1e-4;
    assert(fabs(c->data - (-8.0)) < tol);
    printf("c->data: %.6f, expected: -8.000000\n", c->data);
    assert(fabs(a->grad - (2.0)) < tol);
    printf("a->grad: %.6f, expected: 2.000000\n", a->grad);
    assert(fabs(b->grad - (-4.0)) < tol);
    printf("b->grad: %.6f, expected: -4.000000\n", b->grad);

    printf("test_mul passed\n");
}

void test_pow(void)
{
    Value *a = value_new(-4.0, NULL, 0, "");
    Value *c = value_pow(a, 2.0);
    backward(c);

    float tol = 1e-4;
    assert(fabs(c->data - (16.0)) < tol);
    printf("c->data: %.6f, expected: 16.000000\n", c->data);
    assert(fabs(a->grad - (-8.0)) < tol);
    printf("a->grad: %.6f, expected: -8.000000\n", a->grad);

    printf("test_pow passed\n");
}

void test_relu(void)
{
    Value *a = value_new(-4.0, NULL, 0, "");
    Value *c = value_relu(a);
    backward(c);

    float tol = 1e-4;
    assert(fabs(c->data - (0.0)) < tol);
    printf("c->data: %.6f, expected: 0.000000\n", c->data);
    assert(fabs(a->grad - (0.0)) < tol);
    printf("a->grad: %.6f, expected: 0.000000\n", a->grad);

    printf("test_relu passed\n");
}

void test_neg(void)
{
    Value *a = value_new(-4.0, NULL, 0, "");
    Value *c = value_neg(a);
    backward(c);

    float tol = 1e-4;
    assert(fabs(c->data - (4.0)) < tol);
    printf("c->data: %.6f, expected: 4.000000\n", c->data);
    assert(fabs(a->grad - (-1.0)) < tol);
    printf("a->grad: %.6f, expected: -1.000000\n", a->grad);

    printf("test_neg passed\n");
}

void test_sub(void)
{
    Value *a = value_new(-4.0, NULL, 0, "");
    Value *b = value_new(2.0, NULL, 0, "");
    Value *c = value_sub(a, b);
    backward(c);

    float tol = 1e-4;
    assert(fabs(c->data - (-6.0)) < tol);
    printf("c->data: %.6f, expected: -6.000000\n", c->data);
    assert(fabs(a->grad - (1.0)) < tol);
    printf("a->grad: %.6f, expected: 1.000000\n", a->grad);
    assert(fabs(b->grad - (-1.0)) < tol);
    printf("b->grad: %.6f, expected: -1.000000\n", b->grad);

    printf("test_sub passed\n");
}

void test_div(void)
{
    Value *a = value_new(-4.0, NULL, 0, "");
    Value *b = value_new(2.0, NULL, 0, "");
    Value *c = value_div(a, b);
    backward(c);

    float tol = 1e-4;
    assert(fabs(c->data - (-2.0)) < tol);
    printf("c->data: %.6f, expected: -2.000000\n", c->data);
    assert(fabs(a->grad - (0.5)) < tol);
    printf("a->grad: %.6f, expected: 0.500000\n", a->grad);
    assert(fabs(b->grad - (1.0)) < tol);
    printf("b->grad: %.6f, expected: 1.000000\n", b->grad);

    printf("test_sub passed\n");
}
