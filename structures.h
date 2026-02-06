#ifndef LSH_STRUCTURES_H
#define LSH_STRUCTURES_H
#include <cstdint>
#include <cstdlib>
#include <omp.h>

extern int DOC_SIZE;
extern int DICT_SIZE;
extern int NUM_HASH_FUNCS;
extern int NUM_TABLES;
extern int BAND_SIZE;

#define PRIME 2147483647
#define SET_END -1

struct DenseDoc {
    int* words;
    DenseDoc();
    ~DenseDoc();
    void set_word(int pos, int word_id);
};

struct SetDoc {
    int* words;
    int count;
    SetDoc();
    ~SetDoc();
    void from_dense(const DenseDoc* dense);
};

struct MinHashParams {
    uint32_t* a_params;
    uint32_t* b_params;
    uint32_t* c_params;
    uint32_t* d_params;
    uint32_t prime;
    int num_funcs;
    
    MinHashParams(int num_funcs);
    ~MinHashParams();
    void compute_signature(const SetDoc* doc, uint32_t* signature) const;
    void compute_signature_simd(const SetDoc* doc, uint32_t* signature) const;
};

struct alignas(64) BucketNode {
    int doc_id;
    BucketNode* next;
    char padding[64 - sizeof(int) - sizeof(BucketNode*)];
};

struct alignas(64) AlignedLock {
    omp_lock_t lock;
    char padding[64 - sizeof(omp_lock_t)];
};

struct LSHTable {
    BucketNode** buckets;
    int num_buckets;
    AlignedLock* bucket_locks;
    
    LSHTable(int num_buckets);
    ~LSHTable();
    void insert_with_node(int doc_id, int bucket_id, BucketNode* node);
    void insert_locked_with_node(int doc_id, int bucket_id, BucketNode* node);
};

struct LSHIndex {
    LSHTable** tables;
    int num_tables;
    MinHashParams* minhash;
    int max_docs;
    BucketNode* node_pool;
    int pool_idx;
    int** threads_timestamps;
    int current_query_id;
    int num_threads_allocated;
    
    LSHIndex(int num_tables, int num_hash_funcs, int max_docs, int threads);
    ~LSHIndex();
    void insert(const SetDoc* doc, int doc_id);
    int* query(const SetDoc* doc, int* num_candidates, int tid);
};

uint32_t compute_band_hash(const uint32_t* signature, int band_start, int band_size);
float jaccard_similarity(const SetDoc* doc1, const SetDoc* doc2);

#endif