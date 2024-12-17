#ifndef ENGINE_H
#define ENGINE_H

// a node in the computational graph
typedef struct Value {
    float data;
    float grad;
    void (*_backward)(struct Value *v);
    struct Value **_prev;
    int _prev_count;
    char *_op;
} Value;

// constructor & destructor
Value *value_new(float data, Value **prev, int prev_count, const char *_op);
void value_free(Value *v);

// forward
Value *value_add(Value *a, Value *b);

// backward
void value_add_backward(Value *v);

#endif
