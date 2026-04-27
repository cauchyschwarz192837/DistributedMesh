#pragma once
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <string>
#include <stdexcept>
#include "comm.h"
#include <iostream>
#include "C:/Program Files (x86)/Microsoft SDKs/MPI/Include/mpi.h"

extern int global_step_num;

typedef int DivisionID;
typedef int VertexID;
typedef int WorkerID;
template<class ValueT>
struct BEdge;
template<class MessageT, class HashT>
class MessageBuffer;

struct MeshValue {
    float base_x;
    float base_y;
    float base_z;
    float disp_x;
    float disp_y;
    float disp_z;
    int anchor;
};

struct MeshMessage {
    float dx;
    float dy;
    float dz;
};

template<class T>
void insert_sorted(std::vector<T> &sorted, T &elem) {
    typename std::vector<T>::iterator it = std::lower_bound(sorted.begin(), sorted.end(), elem);
    if(it == sorted.end() || *it != elem) {
        sorted.insert(it, elem); // INSERT GO
    } else {
        *it = elem; // REPLACE ITS
    }
}

template<class T>
typename std::vector<T>::iterator find_sorted_by_id(std::vector<T> &sorted, int new_id) {
    return std::lower_bound(sorted.begin(), sorted.end(), T(new_id));
}

class DefaultHash {
public:
    DivisionID operator()(VertexID id, int num_partitions) {
        if (num_partitions <= 0) {
            throw std::runtime_error("DefaultHash: num_partitions must be > 0");
        }
        return id % num_partitions;
    }
};

//-------------------------------------------------------------------------------------

template<class MessageT>
class BaseCombiner {
public:
	void combine(MessageT &old, MessageT &new_msg) {};
};

template<class T>
class BaseAggregator {
public:
	BaseAggregator() {};

	virtual void aggregate(T &to_be_aggregated) = 0; // override in derived class

	T &value() {
        return aggregated_data;
    };
	
	virtual void reset() {
        value() = T();
    }

	void all_aggregate() {
		int me = get_worker_id();
		int np = get_num_workers();
		if(me == 0) {
			for(int i = 1; i < np; i++) {
				T received = recv_data<T>(i);
				aggregate(received);
			}
			for(int i = 1; i < np; i++) {
				send_data(value(), i);
			}
		}
		else {
			send_data(value(), 0);
			value() = recv_data<T>(0);
		}
	}
private:
	T aggregated_data;
};

template<>
class BaseAggregator<void>{
public:
	void all_aggregate(){};
	virtual void reset(){};
};

//---------------------------------------------------------------------------------------------------------
// Transport conversion

class SerialMe {
public:
    SerialMe() {}
    SerialMe(std::vector<char> b):buf(b) {};
    SerialMe(SerialMe &r) {
        *this = r;
    }

    char* get_buf() {
        return buf.data();
    }
    
    size_t size() {
        return buf.size();
    }

    void raw_byte(char c) {
        buf.push_back(c);
    }

    void raw_bytes(void* ptr, int size) {
        buf.insert(buf.end(), (char*)ptr, (char*)ptr + size);
    }
private:
    std::vector<char> buf;
};

class UnSerialMe { // refactor
public:
    UnSerialMe():index(0) {};
    UnSerialMe(std::vector<char> b):buf(b), index(0) {};
    UnSerialMe(UnSerialMe &r) {
        *this = r;
    }
    UnSerialMe(char* b, size_t s) : buf(b, b + s), index(0) {}

    size_t size() {
        return buf.size();
    }

    size_t position() {
        return index;
    }
    
    char raw_byte() {
        return buf[index++];
    }

    void* raw_bytes(int n_bytes) {
        char* ret = buf.data() + index;
        index += n_bytes;
        return ret;
    }
private:
    std::vector<char> buf;
    size_t index;
};

template<class ValueT>
struct BEdge {
    BEdge(VertexID id, ValueT& v) : target(id), value(v) {}
    BEdge() {}
    VertexID target;
    ValueT value;

    bool operator<(const BEdge<ValueT>& rhs) const {
        return target < rhs.target;
    }

    bool operator==(const BEdge<ValueT>& rhs) const {
        return target == rhs.target;
    }

    bool operator!=(const BEdge<ValueT>& rhs) const {
        return target != rhs.target;
    }
};

template<>
struct BEdge<void> {
    BEdge(int id) : target(id) {}
    BEdge() : target(0) {}

    VertexID target;

    bool operator<(const BEdge<void>& rhs) const {
        return target < rhs.target;
    }

    bool operator==(const BEdge<void>& rhs) const {
        return target == rhs.target;
    }

    bool operator!=(const BEdge<void>& rhs) const {
        return target != rhs.target;
    }
};

inline SerialMe &operator << (SerialMe &m, size_t i) {
    m.raw_bytes(&i, sizeof(size_t));
    return m;
}

inline SerialMe &operator << (SerialMe &m, int i) {
    m.raw_bytes(&i, sizeof(int));
    return m;
}

inline SerialMe &operator << (SerialMe &m, float f) {
    m.raw_bytes(&f, sizeof(float));
    return m;
}

inline UnSerialMe &operator >> (UnSerialMe &m, float& f) {
    f = *(float*)m.raw_bytes(sizeof(float));
    return m;
}

template<class T>
inline SerialMe &operator << (SerialMe &m, T* p) {
    return m << *p;
}

template<class T>
inline SerialMe &operator << (SerialMe &m, std::vector<T> &v) {
    m << v.size();
    for (typename std::vector<T>::iterator it = v.begin(); it != v.end(); ++it) {
        m << *it;
    }
    return m;
}

template<>
inline SerialMe &operator << (SerialMe &m, std::vector<int> &v) {
    m << v.size();
    m.raw_bytes(v.data(), v.size() * sizeof(int));
    return m;
}

template<class T>
inline SerialMe &operator << (SerialMe &m, std::set<T> &v) {
    m << v.size();
    for (typename std::set<T>::iterator it = v.begin(); it != v.end(); ++it) {
        m << *it;
    }
    return m;
}

template<class ValueT>
inline SerialMe &operator << (SerialMe &m, BEdge<ValueT> &e) {
    m << e.target;
    m << e.value;
    return m;
}

template<>
inline SerialMe &operator << (SerialMe &m, BEdge<void> &e) {
    m << e.target;
    return m;
}

template<class ValueT>
inline UnSerialMe &operator >> (UnSerialMe &m, BEdge<ValueT> &e) {
    m >> e.target;
    m >> e.value;
    return m;
}

inline UnSerialMe &operator >> (UnSerialMe &m, size_t &i) {
    i = *(size_t*)m.raw_bytes(sizeof(size_t));
    return m;
}

inline UnSerialMe &operator >> (UnSerialMe &m, int &i) {
    i = *(int*)m.raw_bytes(sizeof(int));
    return m;
}

template<>
inline UnSerialMe &operator >> (UnSerialMe &m, BEdge<void> &e) {
    m >> e.target;
    return m;
}

template<class T>
inline UnSerialMe &operator >> (UnSerialMe &m, T* &p) {
    p = new T;
    return m >> (*p);
}

template<class T>
inline UnSerialMe &operator >> (UnSerialMe &m, std::vector<T> &v) {
    size_t size;
    m >> size;
    v.resize(size);
    for (typename std::vector<T>::iterator it = v.begin(); it != v.end(); it++) {
        m >> *it;
    }
    return m;
}

template<>
inline UnSerialMe &operator >> (UnSerialMe &m, std::vector<int> &v) {
    size_t size;
    m >> size;
    v.resize(size);
    int* data = (int*)m.raw_bytes(sizeof(int) * size);
    v.assign(data, data + size);
    return m;
}

template<class T>
inline UnSerialMe &operator >> (UnSerialMe &m, std::set<T> &v) {
    size_t size;
    m >> size;
    for (size_t i = 0; i < size; i++) {
        T tmp;
        m >> tmp;
        v.insert(v.end(), tmp);
    }
    return m;
}

inline SerialMe& operator<<(SerialMe& m, MeshMessage& msg) {
    m << msg.dx;
    m << msg.dy;
    m << msg.dz;
    return m;
}

inline UnSerialMe& operator>>(UnSerialMe& m, MeshMessage& msg) {
    m >> msg.dx;
    m >> msg.dy;
    m >> msg.dz;
    return m;
}

inline SerialMe& operator<<(SerialMe& m, MeshValue& v) {
    m << v.base_x;
    m << v.base_y;
    m << v.base_z;
    m << v.disp_x;
    m << v.disp_y;
    m << v.disp_z;
    m << v.anchor;
    return m;
}

inline UnSerialMe& operator>>(UnSerialMe& m, MeshValue& v) {
    m >> v.base_x;
    m >> v.base_y;
    m >> v.base_z;
    m >> v.disp_x;
    m >> v.disp_y;
    m >> v.disp_z;
    m >> v.anchor;
    return m;
}

template<class ValueType, class EdgeType, class MessageType, class HashType = DefaultHash, class CombinerType = BaseCombiner<MessageType>, class AggregatorType = BaseAggregator<void> >
class BVertex {
public:
    BVertex() {};
    BVertex(VertexID i):_id(i) {};
    BVertex(VertexID i, ValueType &v):_id(i), _value(v) {};

    friend SerialMe &operator << (SerialMe &m, BVertex &v) {
        m << v._id;
        m << v._value;
        m << v._edges;
        return m;
    }

    friend UnSerialMe &operator >> (UnSerialMe &m, BVertex &v) {
        m >> v._id;
        m >> v._value;
        m >> v._edges;
        return m;
    }

    void compute(std::vector<MessageType> &messages); // MAIN mE!

    ValueType &value() {
        return _value;
    }

    VertexID &id() {
        return _id;
    }

    const VertexID &id() const {
        return _id;
    }

    std::vector<BEdge<EdgeType> > &edges() {
        return _edges;
    }

    void send_message(VertexID id, MessageType &msg) {
        ((MessageBuffer<MessageType, HashType>*)get_message_buffer())->add_message(id, msg);
    }
   
    int step_num() {
        return global_step_num;
    }
    
    AggregatorType &aggregator() {
        return *((AggregatorType*)_get_aggregator());
    }

    bool operator < (const BVertex& rhs) const {
        return _id < rhs._id;
    }

    bool operator == (const BVertex& rhs) const {
        return _id == rhs._id;
    }

    bool operator != (const BVertex& rhs) const {
        return _id != rhs._id;
    }

    void add_edge(BEdge<EdgeType> &edge) {
        insert_sorted(_edges, edge);
    }

private:
    VertexID _id;
    ValueType _value;
    std::vector<BEdge<EdgeType> > _edges;
};

template<class MessageT>
struct IDMsg {
    VertexID id;
    std::vector<MessageT> messages;
    IDMsg() : id(0) {}
    IDMsg(VertexID i, std::vector<MessageT>* m) : id(i), messages(*m) {}

    friend SerialMe &operator << (SerialMe &m, IDMsg<MessageT> &idm) {
        m << idm.id;
        m << idm.messages;
        return m;
    }

    friend UnSerialMe &operator >> (UnSerialMe &m, IDMsg<MessageT> &idm) {
        m >> idm.id;
        m >> idm.messages;
        return m;
    }
};

template<class MessageT, class HashT>
class MessageBuffer { 
public:
    void add_message(VertexID id, MessageT &msg) {
        out_messages[id].push_back(msg);
    }

    std::vector<MessageT> &get_messages(VertexID id) {
        return in_messages[id];
    }

    /*
    clear old inbox
    look at every outgoing destination vertex in out_messages
    figure out which worker owns that vertex
    if it is local, move messages straight into in_messages
    if it is remote, put them into that worker's send bucket
    call bigExchange
    take received remote buckets and append them into in_messages
    clear out_messages
    */

    void sync_messages() {
        int np = get_num_workers();
        int me = get_worker_id();

        in_messages.clear(); // throw away previous round's received messages, build new inbox

        std::vector<std::vector<IDMsg<MessageT> > > to_exchange(np); // each worker has one vector of IDMsg
        for(typename std::unordered_map<VertexID, std::vector<MessageT> >::iterator it = out_messages.begin(); it != out_messages.end(); it++) {
            int vid = it->first;
            int wid = vertex_worker(vid);
            if(wid == me) {
                // my own message, no network exchange needed, just add to in_messages
                in_messages[vid].swap(it->second);
            }
            else {
                to_exchange[wid].push_back(IDMsg<MessageT>(vid, &(it->second)));
            }
        }

        bigExchange(to_exchange); // NETWORK EXCHANGE

        for(int i = 0; i < np; i++) {
            if(i == me) {
                continue;
            }
            std::vector<IDMsg<MessageT> > &messages = to_exchange[i];
            for(int j = 0; j < to_exchange[i].size(); j++) {
                IDMsg<MessageT> &idm = messages[j];
                std::vector<MessageT> &local_m = in_messages[idm.id];
                local_m.insert(local_m.end(), idm.messages.begin(), idm.messages.end());
            }
        }
        out_messages.clear(); // all sent messages done, next round
    }

    void set_pw(std::vector<WorkerID>* pw) {
        partition_worker = pw;
    }

    void set_num_partitions(int n) {
        num_partitions = n;
    }

    int vertex_worker(VertexID id) { // which worker should get it
        return (*partition_worker)[hash(id, num_partitions)]; // look up which worker owns that Division
    }
private:
    HashT hash;
    std::unordered_map<VertexID, std::vector<MessageT> > out_messages;
    std::unordered_map<VertexID, std::vector<MessageT> > in_messages;
    int num_partitions;
    std::vector<WorkerID>* partition_worker;
};

/*
serialize to_exchange[partner] into a SerialMe
send it
receive partner's bytes into UnSerialMe
reconstruct object back into to_exchange[partner]
*/

template<class T>
void bigExchange(std::vector<T> &to_exchange) {
	int np = get_num_workers();
	int me = get_worker_id();

	for(int i = 0; i < np; i++) {
		int partner = (i - me + np) % np; // circular
		if(me != partner) {
			if(me < partner) { // note order send receive, OVERWRITE!!!
				SerialMe m;
				m << to_exchange[partner];
				send_marshall(m, partner);

				UnSerialMe um = recv_unmarshall(partner);
				um >> to_exchange[partner];
			} else {
				UnSerialMe um = recv_unmarshall(partner);
				T received;
				um >> received;

				SerialMe m;
				m << to_exchange[partner];
				send_marshall(m, partner);

				to_exchange[partner] = received;
			}
		}
	}
}

template<class T>
void send_data(const T &data, int dst) {
	SerialMe m;
	m << data;
	send_marshall(m, dst);
}

template<class T>
T recv_data(int src) {
	UnSerialMe um = recv_unmarshall(src);
	T data;
	um >> data;
	return data;
}

//-------------------------------------------------------------------------------------------------------------
// BUILD GRAPH //!TODO

template<class VertexT>
class Division {
    typedef typename VertexT::ValueType ValueType;
    typedef typename VertexT::EdgeType EdgeType;
    typedef typename VertexT::MessageType MessageType;
    typedef typename VertexT::HashType HashType;
public:
    friend inline SerialMe &operator << (SerialMe &m, Division<VertexT> &p) {
        m << p._id;
        m << p._vertexes;
        return m;
    }

    friend inline UnSerialMe &operator >> (UnSerialMe &m, Division<VertexT> &p) {
        m >> p._id;
        m >> p._vertexes;
        return m;
    }

    Division() {}
    Division(DivisionID i) :_id(i) {};
    Division(const Division& rhs) {
        _id = rhs._id;
        _vertexes = rhs._vertexes;
    }

    Division& operator=(const Division& rhs) {
        if (this != &rhs) {
            _id = rhs._id;
            _vertexes = rhs._vertexes;
        }
        return *this;
    }

    void merge_with(const Division& rhs) {
        std::vector<VertexT> new_v(_vertexes.size() + rhs._vertexes.size());
        std::merge(_vertexes.begin(), _vertexes.end(), rhs._vertexes.begin(), rhs._vertexes.end(), new_v.begin());
        _vertexes.swap(new_v);
    }

    bool operator < (const Division<VertexT>& rhs) const {
        return _id < rhs._id;
    }

    bool operator == (const Division<VertexT>& rhs) const {
        return _id == rhs._id;
    }

    bool operator != (const Division<VertexT>& rhs) const {
        return _id != rhs._id;
    }

    DivisionID id() { 
        return _id; 
    }

    DivisionID id() const {
        return _id;
    }

    std::vector<VertexT> &vertexes() {
        return _vertexes; 
    }

    void add_vertex(VertexID id, ValueType &value) {
        VertexT new_vertex(id, value);
        insert_sorted(_vertexes, new_vertex);
    }

    void add_edge(VertexID start, BEdge<EdgeType> &e) {
        typename std::vector<VertexT>::iterator it = find_sorted_by_id(_vertexes, start);
        it->add_edge(e);
    }

    bool halt() {
        return _vertexes.size() == get_halt_count();
    }

    void all_compute() {
        set_halt_count(0);
        for (typename std::vector<VertexT>::iterator vit = _vertexes.begin(); vit != _vertexes.end(); vit++) {
            vit->compute(((MessageBuffer<MessageType, HashType>*)get_message_buffer())->get_messages(vit->id()));
        }
    }

    void clear() {
        _vertexes.clear();
    }
private:
    DivisionID _id;
    std::vector<VertexT> _vertexes;
};

//----------------------------------------------------------------------------------------------------

template<class VertexT>
class GraphBuffer {
    typedef VertexT::EdgeType EdgeType;
public:
    void add_vertex(VertexID id, VertexT::ValueType &v) {
        int part_id = hash(id, num_partitions);
        Division<VertexT> new_p(part_id);
        typename std::vector<Division<VertexT> >::iterator it = lower_bound(all_partitions.begin(), all_partitions.end(), new_p);
        if (it == all_partitions.end() || *it != new_p) {
            it = all_partitions.insert(it, new_p);
        }
        it->add_vertex(id, v);
    }

    void add_edge(VertexID src, BEdge<EdgeType> &e) {
        int part_id = hash(src, num_partitions);
        typename std::vector<Division<VertexT> >::iterator it = find_sorted_by_id(all_partitions, part_id);
        it->add_edge(src, e);
    }

    void sync_graph() {
        int me = get_worker_id();
        int np = get_num_workers();
        std::vector<std::vector<Division<VertexT>* > > to_exchange(np);
        for(typename std::vector<Division<VertexT> >::iterator it = all_partitions.begin(); it != all_partitions.end(); it++) {
            int wid = (*partition_worker)[it->id()];
            if(wid == me) {
                _my_partitions.push_back(*it);
            }
            else {
                to_exchange[wid].push_back(&(*it));
            }
        }

        bigExchange(to_exchange);

        for(int i = 0; i < np; i++) {
            if(i == me) {
                continue;
            }
            std::vector<Division<VertexT>*> &parts = to_exchange[i];
            for(int j = 0; j < parts.size(); j++) {
                Division<VertexT> &p = *(parts[j]);
                typename std::vector<Division<VertexT> >::iterator it = lower_bound(_my_partitions.begin(), _my_partitions.end(), p);
                if (it != _my_partitions.end() && *it == p) {
                    it->merge_with(p);
                } else {
                    _my_partitions.insert(it, p);
                }
            }
        }
        all_partitions.clear();
    };

    std::vector<Division<VertexT> > &my_partitions() {
        return _my_partitions; 
    }

    void set_pw(std::vector<WorkerID>* pw) { 
        partition_worker = pw; 
    }

    void set_num_partitions(int n) { 
        num_partitions = n; 
    }
private:
    VertexT::HashType hash;
    int num_partitions;
    std::vector<Division<VertexT> > _my_partitions;
    std::vector<Division<VertexT> > all_partitions;
    std::vector<WorkerID>* partition_worker;
};

//---------------------------------------------------------------------------------------

template<class VertexT, class GraphLoaderT>
class Worker {
    typedef typename VertexT::MessageType MessageT;
    typedef typename VertexT::HashType HashT;
    typedef typename VertexT::AggregatorType AggregatorT;
public:
    Worker() {
        id = get_worker_id();
        num_workers = get_num_workers();
        set_message_buffer(&message_buffer);
        _set_aggregator(&_aggregator);
    }

    std::vector<Division<VertexT>>& local_partitions() {
        return partitions;
    }

    void run(int num_partitions) {
        if (num_workers <= 0) {
            throw std::runtime_error("Worker::run: num_workers must be > 0. Did you call init_pregel()?");
        }

        if (num_partitions <= 0) {
            throw std::runtime_error("Worker::run: num_partitions must be > 0.");
        }

        partition_worker.resize(num_partitions);
        for (int i = 0; i < num_partitions; i++) {
            partition_worker[i] = i % num_workers;
        }
        
        message_buffer.set_pw(&partition_worker);
        message_buffer.set_num_partitions(num_partitions);

        GraphBuffer<VertexT> graph_buffer;
        graph_buffer.set_num_partitions(num_partitions);
        graph_buffer.set_pw(&partition_worker);
        GraphLoaderT loader;
        loader.set_id(id);
        loader.set_buffer(&graph_buffer);
        loader.load_graph();

        std::cout << "[rank " << get_worker_id() << "] before sync_graph\n";
        graph_buffer.sync_graph();
        std::cout << "[rank " << get_worker_id() << "] after sync_graph\n";
        partitions.swap(graph_buffer.my_partitions());

        int local_partition_count = static_cast<int>(partitions.size());
        std::cout << "[rank " << get_worker_id() << "] before all_sum Division count\n";
        int global_partition_count = all_sum(local_partition_count);
        std::cout << "[rank " << get_worker_id() << "] after all_sum Division count\n";
        std::cout << "[rank " << get_worker_id() << "] before barrier\n";
        MPI_Barrier(MPI_COMM_WORLD);
        std::cout << "[rank " << get_worker_id() << "] after barrier\n";

        bool halt = false;
        int step_num = 0;

        while (!halt) {
            global_step_num = step_num; // vertices can access
            halt_partition_count = 0;
            _aggregator.reset();

            for (typename std::vector<Division<VertexT> >::iterator it = partitions.begin(); it != partitions.end(); it++) {
                it->all_compute();
                if (it->halt()) {
                    halt_partition_count++;
                }
            }

            message_buffer.sync_messages();
            _aggregator.all_aggregate(); // each worker has a local value, all_aggregate() combines them globally.
            step_num++;

            int all_halt_count = 0;
            all_halt_count = all_sum(halt_partition_count); // each worker knows only its own halted partitions
            if (all_halt_count == global_partition_count) {
                halt = true;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD); // all workers finish the loop
    }
private:
    WorkerID id;
    int num_workers;
    int num_partitions;
    int halt_partition_count;
    std::vector<WorkerID> partition_worker;
    std::vector<Division<VertexT> > partitions;
    MessageBuffer<MessageT, HashT> message_buffer;
    AggregatorT _aggregator;
};
