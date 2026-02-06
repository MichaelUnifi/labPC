#include "structures.h"
#include <cstdio>
#include <ctime>
#include <vector>
#include <algorithm>

#define NUM_INSERT_DOCS 100000
#define NUM_QUERY_DOCS 100000
#define RUNS_PER_EXPERIMENT 10

struct ExperimentConfig {
    int doc_size;
    int dict_size;
    int band_size;
    int num_tables;
    const char* name;
    const char* scenario;
};

struct RunResults {
    double insertion_time;
    double query_time;
    int total_candidates;
    int total_true_positives;
    double avg_jaccard_all_pairs;
    double avg_jaccard_candidates;
};

void generate_document(DenseDoc* doc, int doc_id, int dict_size) {

    unsigned int seed = doc_id * 12345;
    int max_common = dict_size / 170;
    int max_medium = dict_size / 17;
    int max_rare = dict_size / 4;

    for (int i = 0; i < DOC_SIZE; i++) {
        float r = static_cast<float>(rand_r(&seed)) / RAND_MAX;
        if (r < 0.5f)
            doc->set_word(i, (rand_r(&seed) % max_common) + 1);
        else if (r < 0.8f)
            doc->set_word(i, (rand_r(&seed) % (max_medium - max_common)) + max_common + 1);
        else if (r < 0.95f)
            doc->set_word(i, (rand_r(&seed) % (max_rare - max_medium)) + max_medium + 1);
        else
            doc->set_word(i, (rand_r(&seed) % (dict_size - max_rare)) + max_rare + 1);
    }
} 

RunResults run_single_experiment(SetDoc** set_docs, SetDoc** query_docs) {
    RunResults results = {0};
    clock_t start, end;

    LSHIndex* index = new LSHIndex(NUM_TABLES, NUM_HASH_FUNCS, NUM_INSERT_DOCS, 1);

    start = clock();
    for (int i = 0; i < NUM_INSERT_DOCS; i++) {
        index->insert(set_docs[i], i);
    }
    end = clock();
    results.insertion_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;

    int total_candidates = 0;
    int total_true_positives = 0;
    float similarity_threshold = 0.5f;
    
    double total_jaccard_candidates = 0.0;
    int jaccard_candidate_count = 0;

    start = clock();
    for (int i = 0; i < NUM_QUERY_DOCS; i++) {
        int num_candidates;
        int* candidates = index->query(query_docs[i], &num_candidates, 0);
        num_candidates = num_candidates >= 50 ? 50 : num_candidates;
        total_candidates += num_candidates;

        for (int j = 0; j < num_candidates; j++) {
            float sim = jaccard_similarity(query_docs[i], set_docs[candidates[j]]);
            total_jaccard_candidates += sim;
            jaccard_candidate_count++;
            
            if (sim >= similarity_threshold) {
                total_true_positives++;
            }
        }

        delete[] candidates;
    }
    end = clock();
    results.query_time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    results.total_candidates = total_candidates;
    results.total_true_positives = total_true_positives;
    results.avg_jaccard_candidates = jaccard_candidate_count > 0 ? 
        total_jaccard_candidates / jaccard_candidate_count : 0.0;

    double total_jaccard_random = 0.0;
    int sample_size = 100;
    for (int i = 0; i < sample_size; i++) {
        int query_idx = rand() % NUM_QUERY_DOCS;
        int insert_idx = rand() % NUM_INSERT_DOCS;
        total_jaccard_random += jaccard_similarity(query_docs[query_idx], set_docs[insert_idx]);
    }
    results.avg_jaccard_all_pairs = total_jaccard_random / sample_size;

    delete index;
    return results;
}

void run_scaling_benchmark() {
    printf("=== LSH Bit-Signature Sequential Scaling Benchmark ===\n\n");

    std::vector<ExperimentConfig> configs = {
        {200,   170000, 20, 5, "size_small",  "Small (200 words)"},
        {500,   170000, 20, 5, "size_medium", "Medium (500 words)"},
        {1000,  170000, 20, 5, "size_large",  "Large (1000 words)"},
        {3000,  170000, 20, 5, "size_xlarge",  "Extra Large (3k words)"},
        {500,   170000, 20,  5, "tables_5",    "Tables scaling (5)"},
        {500,   170000, 20, 10, "tables_10",   "Tables scaling (10)"},
        {500,   170000, 20, 15, "tables_15",   "Tables scaling (15)"},
        {500,   170000, 20, 5, "band_20",     "Band size scaling (20)"},
        {500,   170000, 25, 5, "band_25",     "Band size scaling (25)"},
        {500,   170000, 30, 5, "band_30",     "Band size scaling (30)"},
    };

    FILE* json_file = fopen("results_sequential_scaling.json", "w");
    fprintf(json_file, "{\n");
    fprintf(json_file, "  \"benchmark\": \"sequential_scaling_bit_lsh\",\n");
    fprintf(json_file, "  \"num_insert_docs\": %d,\n", NUM_INSERT_DOCS);
    fprintf(json_file, "  \"num_query_docs\": %d,\n", NUM_QUERY_DOCS);
    fprintf(json_file, "  \"runs_per_experiment\": %d,\n", RUNS_PER_EXPERIMENT);
    fprintf(json_file, "  \"similarity_threshold\": 0.5,\n");
    fprintf(json_file, "  \"results\": [\n");

    bool first_config = true;
    for (const auto& config : configs) {
        DOC_SIZE       = config.doc_size;
        DICT_SIZE      = config.dict_size;
        BAND_SIZE      = config.band_size;
        NUM_TABLES     = config.num_tables;
        NUM_HASH_FUNCS = BAND_SIZE * NUM_TABLES;

        printf("=== Configuration: %s ===\n", config.name);
        printf("  Doc size: %d, Dict size: %d, Hash funcs: %d, Tables: %d, Band size: %d\n",
            DOC_SIZE, DICT_SIZE, NUM_HASH_FUNCS, NUM_TABLES, BAND_SIZE);
        const int VARIANTS_PER_SEED = 50; 
        const int NUM_SEEDS = NUM_INSERT_DOCS / VARIANTS_PER_SEED;

        DenseDoc** dense_docs = new DenseDoc*[NUM_INSERT_DOCS];
        SetDoc** set_docs   = new SetDoc*[NUM_INSERT_DOCS];

        printf("Generating %d insert docs (%d clusters of %d)...\n", NUM_INSERT_DOCS, NUM_SEEDS, VARIANTS_PER_SEED);

        for (int s = 0; s < NUM_SEEDS; s++) {
            int seed_idx = s * VARIANTS_PER_SEED;
            dense_docs[seed_idx] = new DenseDoc();
            generate_document(dense_docs[seed_idx], s + 1000, DICT_SIZE);
            
            set_docs[seed_idx] = new SetDoc();
            set_docs[seed_idx]->from_dense(dense_docs[seed_idx]);

            for (int v = 1; v < VARIANTS_PER_SEED; v++) {
                int current_idx = seed_idx + v;
                dense_docs[current_idx] = new DenseDoc();

                std::copy(dense_docs[seed_idx]->words, 
                        dense_docs[seed_idx]->words + DOC_SIZE, 
                        dense_docs[current_idx]->words);
                
                unsigned int variant_seed = current_idx * 12345;
                int num_mutations = DOC_SIZE / 14; 
                for (int m = 0; m < num_mutations; m++) {
                    int pos = rand_r(&variant_seed) % DOC_SIZE;
                    int stop_word_boundary = DICT_SIZE / 170;
                    int new_word = (rand_r(&variant_seed) % (DICT_SIZE - stop_word_boundary)) + stop_word_boundary;
                    dense_docs[current_idx]->set_word(pos, new_word);
                }
                set_docs[current_idx] = new SetDoc();
                set_docs[current_idx]->from_dense(dense_docs[current_idx]);
            }
        }

        DenseDoc** query_dense = new DenseDoc*[NUM_QUERY_DOCS];
        SetDoc** query_docs  = new SetDoc*[NUM_QUERY_DOCS];

        printf("Generating %d query docs (mutated versions of insert docs)...\n", NUM_QUERY_DOCS);

        for (int i = 0; i < NUM_QUERY_DOCS; i++) {
            query_dense[i] = new DenseDoc();
            int target_family_seed_idx = (i % NUM_SEEDS) * VARIANTS_PER_SEED;
            std::copy(dense_docs[target_family_seed_idx]->words, 
                dense_docs[target_family_seed_idx]->words + DOC_SIZE, 
                query_dense[i]->words);
            
            unsigned int query_seed = i + 99999;
            int num_mutations = DOC_SIZE / 10; 
            for (int m = 0; m < num_mutations; m++) {
                int pos = rand_r(&query_seed) % DOC_SIZE;
                int stop_word_boundary = DICT_SIZE / 170;
                int new_word = (rand_r(&query_seed) % (DICT_SIZE - stop_word_boundary)) + stop_word_boundary;
                query_dense[i]->set_word(pos, new_word);
            }
            
            query_docs[i] = new SetDoc();
            query_docs[i]->from_dense(query_dense[i]);
        }

        double avg_insertion = 0, avg_query = 0, avg_candidates = 0;
        double avg_precision = 0, avg_recall_estimate = 0;
        double avg_jaccard_all = 0, avg_jaccard_cand = 0;

        for (int run = 0; run < RUNS_PER_EXPERIMENT; run++) {
            printf("  Run %d/%d...\n", run + 1, RUNS_PER_EXPERIMENT);
            RunResults res = run_single_experiment(set_docs, query_docs);

            avg_insertion += res.insertion_time;
            avg_query     += res.query_time;
            avg_candidates += res.total_candidates;
            avg_jaccard_all += res.avg_jaccard_all_pairs;
            avg_jaccard_cand += res.avg_jaccard_candidates;
            
            double precision = res.total_candidates > 0 ?
                static_cast<double>(res.total_true_positives) / res.total_candidates : 0.0;
            avg_precision += precision;

            double estimated_total_tp = res.avg_jaccard_all_pairs * NUM_INSERT_DOCS;
            double recall = estimated_total_tp > 0 ? 
                res.total_true_positives / (estimated_total_tp * NUM_QUERY_DOCS) : 0.0;
            avg_recall_estimate += recall;
        }

        avg_insertion  /= RUNS_PER_EXPERIMENT;
        avg_query      /= RUNS_PER_EXPERIMENT;
        avg_candidates /= RUNS_PER_EXPERIMENT;
        avg_precision  /= RUNS_PER_EXPERIMENT;
        avg_recall_estimate /= RUNS_PER_EXPERIMENT;
        avg_jaccard_all /= RUNS_PER_EXPERIMENT;
        avg_jaccard_cand /= RUNS_PER_EXPERIMENT;

        printf("  Avg insertion: %.4fs, Avg query: %.4fs\n", avg_insertion, avg_query);
        printf("  Avg candidates/query: %.2f\n", avg_candidates / NUM_QUERY_DOCS);
        printf("  Avg precision: %.4f, Estimated recall: %.4f\n", avg_precision, avg_recall_estimate);
        printf("  Avg Jaccard (random pairs): %.4f, (candidates): %.4f\n\n", 
               avg_jaccard_all, avg_jaccard_cand);

        if (!first_config) fprintf(json_file, ",\n");
        first_config = false;

        fprintf(json_file, "    {\n");
        fprintf(json_file, "      \"name\": \"%s\",\n", config.name);
        fprintf(json_file, "      \"doc_size\": %d,\n", config.doc_size);
        fprintf(json_file, "      \"dict_size\": %d,\n", config.dict_size);
        fprintf(json_file, "      \"num_tables\": %d,\n", config.num_tables);
        fprintf(json_file, "      \"band_size\": %d,\n", BAND_SIZE);
        fprintf(json_file, "      \"avg_insertion_time\": %.6f,\n", avg_insertion);
        fprintf(json_file, "      \"avg_query_time\": %.6f,\n", avg_query);
        fprintf(json_file, "      \"avg_candidates_per_query\": %.2f,\n", avg_candidates / NUM_QUERY_DOCS);
        fprintf(json_file, "      \"avg_precision\": %.4f,\n", avg_precision);
        fprintf(json_file, "      \"estimated_recall\": %.4f,\n", avg_recall_estimate);
        fprintf(json_file, "      \"avg_jaccard_random_pairs\": %.4f,\n", avg_jaccard_all);
        fprintf(json_file, "      \"avg_jaccard_candidates\": %.4f\n", avg_jaccard_cand);
        fprintf(json_file, "    }");

        for (int i = 0; i < NUM_INSERT_DOCS; i++) { 
            delete dense_docs[i]; 
            delete set_docs[i]; 
        }
        delete[] dense_docs; 
        delete[] set_docs;
        
        for (int i = 0; i < NUM_QUERY_DOCS; i++) { 
            delete query_dense[i]; 
            delete query_docs[i]; 
        }
        delete[] query_dense; 
        delete[] query_docs;
    }

    fprintf(json_file, "\n  ]\n}\n");
    fclose(json_file);
    printf("Results saved to results_sequential_scaling.json\n");
}

int main() {
    run_scaling_benchmark();
    printf("\n=== Benchmark completed! ===\n");
    return 0;
}