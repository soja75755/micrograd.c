#include "engine.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Value *value_new(float data, Value **prev, int prev_count, const char *op)
{
    Value *v = malloc(sizeof(Value));
    if (v == NULL) {
        fprintf(stderr, "Fail to allocate memory\n");
        return NULL;
    }
    v->data = data;
    v->grad = .0;
    v->_backward = NULL;
    v->_prev = malloc(prev_count * sizeof(Value *));
    memcpy(v->_prev, prev, prev_count * sizeof(Value *));
    v->_prev_count = prev_count;
    v->_op = strdup(op);
    return v;
}

void value_free(Value *v)
{
    free(v->_prev);
    free(v->_op);
    free(v);
}

Value *value_add(Value *a, Value *b)
{
    Value *prev[] = {a, b};
    Value *out = value_new(a->data + b->data, prev, 2, "+");

    out->_backward = value_add_backward;

    return out;
}


Value *value_mul(Value *a, Value *b)
{
    Value *prev[] = {a, b};
    Value *out = value_new(a->data * b->data, prev, 2, "*");

    out->_backward = value_mul_backward;

    return out;
}


Value *value_pow(Value *a, float power)
{
    Value *prev[] = {a};
    char power_str[20];
    snprintf(power_str, sizeof(power_str), "**%.2f", power);
    Value *out = value_new(powf(a->data, power), prev, 1, power_str);

    out->_backward = value_pow_backward;

    return out;
}

Value *value_relu(Value *a)
{
    Value *prev[] = {a};
    Value *out = value_new(a->data > 0 ? a->data : 0, prev, 1, "Relu");

    out->_backward = value_relu_backward;

    return out;
}

void value_add_backward(Value *v)
{
    v->_prev[0]->grad += v->grad;
    v->_prev[1]->grad += v->grad;
}

void value_mul_backward(Value *v)
{
    v->_prev[0]->grad += v->_prev[1]->data * v->grad;
    v->_prev[1]->grad += v->_prev[0]->data * v->grad;
}

void value_pow_backward(Value *v)
{
    float power;
    sscanf(v->_op + 2, "%f", &power);
    v->_prev[0]->grad += power * powf(v->_prev[0]->data, power - 1) * v->grad;
}

void value_relu_backward(Value *v)
{
    v->_prev[0]->grad += (v->_prev[0]->data > 0) ? v->grad : 0;
}

Value *value_neg(Value *a)
{
    return value_mul(a, value_new(-1, NULL, 0, ""));
}

Value *value_sub(Value *a, Value *b)
{
    return value_add(a, value_neg(b));
}

Value *value_div(Value *a, Value *b)
{
    return value_mul(a, value_pow(b, -1));
}

void backward(Value *v)
{
    // gradient
    v->grad = 1;

    if (v->_backward) {
        v->_backward(v);
    }
}
