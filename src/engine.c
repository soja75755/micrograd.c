#include "engine.h"
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

void value_add_backward(Value *v)
{
    v->_prev[0]->grad += v->grad;
    v->_prev[1]->grad += v->grad;
}
