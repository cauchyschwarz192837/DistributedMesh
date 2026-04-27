#pragma once
#include <cstddef>

class Marshall;
class Unmarshall;

int get_worker_id();
int get_num_workers();

void init_workers(int* argc, char*** argv);
int all_sum(int my_copy);

void pregel_send(void* buf, int size, int dst);
void pregel_recv(void* buf, int size, int src);

void send_marshall(Marshall& m, int dst);
Unmarshall recv_unmarshall(int src);

template<class T>
void send_data(const T & data, int dst);

template<class T>
T recv_data(int src);

void set_message_buffer(void* mb);
void* get_message_buffer();

void set_graph_buffer(void* gb);
void* get_graph_buffer();

void set_halt_count(int i);
int get_halt_count();
void vote_for_halt();

void _set_aggregator(void* ptr);
void* _get_aggregator();

void init_pregel(int argc, char** argv);

extern int global_step_num;