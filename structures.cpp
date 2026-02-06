#include "structures.h"
#include <cstring>
#include <cstdio>
#include <ctime>
#include <algorithm>
#include <omp.h>
#include <immintrin.h>

int DOC_SIZE = 80000;
int DICT_SIZE = 1000000;
int NUM_HASH_FUNCS = 200;
int NUM_TABLES = 10;
int BAND_SIZE = 20;


DenseDoc::DenseDoc() {
    words = new int[DOC_SIZE];
    std::memset(words, 0, DOC_SIZE * sizeof(int));
}

DenseDoc::~DenseDoc() {
    delete[] words;
}

void DenseDoc::set_word(int pos, int word_id) {
    words[pos] = word_id;
}

SetDoc::SetDoc() : words(nullptr), count(0) {}

SetDoc::~SetDoc() {
    delete[] words;
}

void SetDoc::from_dense(const DenseDoc* dense) {
    int* temp = new int[DOC_SIZE];
    int temp_count = 0;

    for (int i = 0; i < DOC_SIZE; i++) {
        if (dense->words[i] != 0) {
            temp[temp_count++] = dense->words[i];
        }
    }
    std::sort(temp, temp + temp_count);
    count = 0;
    for (int i = 0; i < temp_count; i++) {
        if (i == 0 || temp[i] != temp[i - 1]) count++;
    }
    words = new int[count + 1];
    int idx = 0;
    for (int i = 0; i < temp_count; i++) {
        if (i == 0 || temp[i] != temp[i - 1]) {
            words[idx++] = temp[i];
        }
    }
    words[count] = SET_END;

    delete[] temp;
}

MinHashParams::MinHashParams(int num_funcs) : num_funcs(num_funcs), prime(PRIME) {
    a_params = new uint32_t[num_funcs];
    b_params = new uint32_t[num_funcs];
    c_params = new uint32_t[num_funcs];
    d_params = new uint32_t[num_funcs];

    srand(time(nullptr));
    for (int i = 0; i < num_funcs; i++) {
        a_params[i] = (rand() % (PRIME - 1)) + 1;
        b_params[i] = rand() % PRIME;
        c_params[i] = (rand() % (PRIME - 1)) + 1;
        d_params[i] = rand() % PRIME;
    }
}

MinHashParams::~MinHashParams() {
    delete[] a_params;
    delete[] b_params;
    delete[] c_params;
    delete[] d_params;
}

void MinHashParams::compute_signature_simd(const SetDoc* doc, uint32_t* signature) const {
    const int num_f = this->num_funcs;
    const int* words = doc->words;
    const uint32_t prime_val = 0x7FFFFFFF;

    int j = 0;
    __m256i prime_v = _mm256_set1_epi32(prime_val);
    for (; j <= num_f - 8; j += 8) {
        _mm256_storeu_si256((__m256i*)&signature[j], prime_v);
    }
    for (; j < num_f; j++) {
        signature[j] = prime_val;
    }

    __m256i mask_mersenne = _mm256_set1_epi32(0x7FFFFFFF);
    for (int i = 0; words[i] != SET_END; i++) {
        uint32_t word = static_cast<uint32_t>(words[i]);
        __m256i word_v = _mm256_set1_epi32(word);

        j = 0;
        for (; j <= num_f - 8; j += 8) {
            __m256i a_v = _mm256_loadu_si256((const __m256i*)&a_params[j]);
            __m256i b_v = _mm256_loadu_si256((const __m256i*)&b_params[j]);
            __m256i h_v = _mm256_add_epi32(_mm256_mullo_epi32(a_v, word_v), b_v);
            h_v = _mm256_add_epi32(_mm256_and_si256(h_v, mask_mersenne), _mm256_srli_epi32(h_v, 31));
            __m256i overflow = _mm256_cmpgt_epi32(h_v, mask_mersenne);
            h_v = _mm256_sub_epi32(h_v, _mm256_and_si256(overflow, prime_v));
            __m256i min_v = _mm256_loadu_si256((__m256i*)&signature[j]);
            min_v = _mm256_min_epu32(h_v, min_v);
            _mm256_storeu_si256((__m256i*)&signature[j], min_v);
        }
        for (; j < num_f; j++) {
            uint32_t h = (a_params[j] * word + b_params[j]);
            h = (h & 0x7FFFFFFF) + (h >> 31);
            if (h >= prime_val) h -= prime_val;
            if (h < signature[j]) signature[j] = h;
        }
    }
    j = 0;
    for (; j <= num_f - 8; j += 8) {
        __m256i c_v = _mm256_loadu_si256((const __m256i*)&c_params[j]);
        __m256i d_v = _mm256_loadu_si256((const __m256i*)&d_params[j]);
        __m256i min_v = _mm256_loadu_si256((__m256i*)&signature[j]);
        __m256i h2_v = _mm256_add_epi32(_mm256_mullo_epi32(c_v, min_v), d_v);
        h2_v = _mm256_add_epi32(_mm256_and_si256(h2_v, mask_mersenne), _mm256_srli_epi32(h2_v, 31));
        __m256i overflow = _mm256_cmpgt_epi32(h2_v, mask_mersenne);
        h2_v = _mm256_sub_epi32(h2_v, _mm256_and_si256(overflow, prime_v));
        __m256i res_v = _mm256_and_si256(h2_v, _mm256_set1_epi32(1));
        _mm256_storeu_si256((__m256i*)&signature[j], res_v);
    }
    for (; j < num_f; j++) {
        uint32_t h2 = (c_params[j] * signature[j] + d_params[j]);
        h2 = (h2 & 0x7FFFFFFF) + (h2 >> 31);
        if (h2 >= prime_val) h2 -= prime_val;
        signature[j] = h2 & 1;
    }
}

LSHTable::LSHTable(int num_buckets) : num_buckets(num_buckets) {
    buckets = new BucketNode*[num_buckets]();
    bucket_locks = new AlignedLock[num_buckets];
    for (int i = 0; i < num_buckets; i++)
        omp_init_lock(&bucket_locks[i].lock);
}

LSHTable::~LSHTable() {
    for (int i = 0; i < num_buckets; i++)
        omp_destroy_lock(&bucket_locks[i].lock);
    delete[] bucket_locks;
    delete[] buckets;
}

void LSHTable::insert_with_node(int doc_id, int bucket_id, BucketNode* node) {
    node->doc_id = doc_id;
    node->next = buckets[bucket_id];
    buckets[bucket_id] = node;
}

void LSHTable::insert_locked_with_node(int doc_id, int bucket_id, BucketNode* node) {
    omp_set_lock(&bucket_locks[bucket_id].lock);
    node->doc_id = doc_id;
    node->next = buckets[bucket_id];
    buckets[bucket_id] = node;
    omp_unset_lock(&bucket_locks[bucket_id].lock);
}

LSHIndex::LSHIndex(int num_tables, int num_hash_funcs, int max_docs, int threads) 
    : num_tables(num_tables), max_docs(max_docs), pool_idx(0), current_query_id(0), num_threads_allocated(threads) {
    
    minhash = new MinHashParams(num_hash_funcs);
    tables = new LSHTable*[num_tables];
    for (int i = 0; i < num_tables; i++) 
        tables[i] = new LSHTable(1 << 20);
    node_pool = new BucketNode[max_docs * num_tables];
    
    threads_timestamps = new int*[num_threads_allocated];
    for (int i = 0; i < num_threads_allocated; i++) {
        threads_timestamps[i] = new int[max_docs]();
    }
}

LSHIndex::~LSHIndex() {
    for (int i = 0; i < num_tables; i++) delete tables[i];
    delete[] tables;
    delete minhash;
    delete[] node_pool;
    for (int i = 0; i < num_threads_allocated; i++) {
        delete[] threads_timestamps[i];
    }
    delete[] threads_timestamps;
}

void LSHIndex::insert(const SetDoc* doc, int doc_id) {
    uint32_t* signature = new uint32_t[NUM_HASH_FUNCS];
    minhash->compute_signature_simd(doc, signature);
    int band_size = NUM_HASH_FUNCS / num_tables;

    for (int i = 0; i < num_tables; i++) {
        uint32_t band_hash = compute_band_hash(signature, i * band_size, band_size);
        int bucket_idx = band_hash % tables[i]->num_buckets;
        
        int idx;
        #pragma omp atomic capture
        idx = pool_idx++;
        tables[i]->insert_with_node(doc_id, bucket_idx, &node_pool[idx]);
    }
    
    delete[] signature;
}

int* LSHIndex::query(const SetDoc* doc, int* num_candidates, int tid) {
    uint32_t* signature = new uint32_t[NUM_HASH_FUNCS];
    minhash->compute_signature_simd(doc, signature);
    
    int* my_timestamps = threads_timestamps[tid];
    int q_id;
    #pragma omp atomic capture
    q_id = ++current_query_id;
    
    int band_size = NUM_HASH_FUNCS / num_tables;
    int* candidates = new int[max_docs];
    int count = 0;

    for (int t = 0; t < num_tables; t++) {
        uint32_t band_hash = compute_band_hash(signature, t * band_size, band_size);
        int bucket_idx = band_hash % tables[t]->num_buckets;
        BucketNode* node = tables[t]->buckets[bucket_idx];

        while (node) {
            if (my_timestamps[node->doc_id] < q_id) {
                my_timestamps[node->doc_id] = q_id;
                candidates[count++] = node->doc_id;
            }
            node = node->next;
        }
    }
    
    *num_candidates = count;
    delete[] signature;
    return candidates;
}

uint32_t compute_band_hash(const uint32_t* signature, int band_start, int band_size) {
    uint32_t hash = 0;
    for (int i = 0; i < band_size; i++) {
        hash = (hash << 1) | signature[band_start + i];
    }
    return hash;
}

float jaccard_similarity(const SetDoc* __restrict doc1, const SetDoc* __restrict doc2) {
    const int *p1 = doc1->words;
    const int *p2 = doc2->words;
    const int *end1 = p1 + doc1->count;
    const int *end2 = p2 + doc2->count;
    int intersection = 0;
    while (p1 < end1 && p2 < end2) {
        if (*p1 == *p2) { ++intersection; ++p1; ++p2; }
        else if (*p1 < *p2) ++p1;
        else ++p2;
    }
    int union_size = doc1->count + doc2->count - intersection;
    return union_size > 0 ? (float)intersection / (float)union_size : 0.0f;
}