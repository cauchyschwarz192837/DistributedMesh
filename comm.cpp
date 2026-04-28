#include "comm.h"
#include "main.h"
#include "C:/Program Files (x86)/Microsoft SDKs/MPI/Include/mpi.h"

static int _my_rank, _num_processors; // at file scope internal link
void* global_message_buffer = nullptr;
void* global_graph_buffer = nullptr;
void* _global_aggregator = nullptr;
int global_step_num;

int get_worker_id() {
	return _my_rank;
}

int get_num_workers() {
	return _num_processors;
}

void init_workers(int* argc, char*** argv ) {
	MPI_Init(argc, argv);
	MPI_Comm_size(MPI_COMM_WORLD, &_num_processors);
	MPI_Comm_rank(MPI_COMM_WORLD, &_my_rank); // ASSIGN!!!!!!!
}

int all_sum(int my_copy) { // every worker contributs one integer, and everyone gets back the sum across all workers
	int toRet = 0;
	MPI_Allreduce(&my_copy, &toRet, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	return toRet;
}

void pregel_send(void* buf, int size, int dst) { // send size bytes to destination worker dst
	MPI_Send((char*)buf, size, MPI_CHAR, dst, 0, MPI_COMM_WORLD);
}

void pregel_recv(void* buf, int size, int src) { // receive size bytes from source worker src
	MPI_Recv(static_cast<char*>(buf), size, MPI_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void send_marshall(SerialMe &m, int dst) {
	size_t size = m.size();
	pregel_send(&size, sizeof(size_t), dst);
	pregel_send(m.get_buf(), m.size(), dst); // send the size and the actual buffer
}

/*
receive the size
allocate a byte buffer of that size
receive the actual bytes
wrap them in an UnSerialMe
*/

UnSerialMe recv_unmarshall(int src) {
    size_t size;
    pregel_recv(&size, sizeof(size_t), src);

    std::vector<char> buf(size);
    pregel_recv(buf.data(), static_cast<int>(size), src);

    return UnSerialMe(buf); // declared constructor
}

void set_message_buffer(void* mb) {
	global_message_buffer = mb;
}

void* get_message_buffer() {
	return global_message_buffer;
}

// graph buffer

void set_graph_buffer(void* gb) {
	global_graph_buffer = gb;
}

void* get_graph_buffer() {
	return global_graph_buffer;
}

// global data

int global_halt_count = 0;

void set_halt_count(int i) {
	global_halt_count = i;
}

int get_halt_count() {
	return global_halt_count;
}

void vote_for_halt() {
	global_halt_count++;
}

int step_num(){
	return global_step_num;
}

void _set_aggregator(void* ptr) {
	_global_aggregator = ptr;
}

void* _get_aggregator() {
	return _global_aggregator;
}
