

#define PREDICT 1
#define UPDATE 0

struct TsetlinMachine { 
	int number_of_clauses;

	int number_of_features;

	int number_of_clause_chunks;

	omp_lock_t *clause_lock;

	unsigned int *ta_state;
	unsigned int *clause_output;
	unsigned int *feedback_to_la;
	int *feedback_to_clauses;
	unsigned int *clause_patch;

	int *output_one_patches;

	unsigned int *clause_weights;

	int number_of_patches;
	int number_of_ta_chunks;
	int number_of_state_bits;

	int T;

	double s;

	double s_range;

	unsigned int filter;

	int boost_true_positive_feedback;

	int weighted_clauses;
};

struct TsetlinMachine *CreateTsetlinMachine(int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range, int boost_true_positive_feedback, int weighted_clauses);

void tm_initialize(struct TsetlinMachine *tm);

void tm_destroy(struct TsetlinMachine *tm);

void tm_update_clauses(struct TsetlinMachine *tm, unsigned int *Xi, int class_sum, int target);

void tm_update(struct TsetlinMachine *tm, unsigned int *Xi, int target);

int tm_score(struct TsetlinMachine *tm, unsigned int *Xi);

int tm_ta_state(struct TsetlinMachine *tm, int clause, int ta);

int tm_ta_action(struct TsetlinMachine *tm, int clause, int ta);

void tm_update_regression(struct TsetlinMachine *tm, unsigned int *Xi, int target);

void tm_fit_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples, int epochs);

int tm_score_regression(struct TsetlinMachine *tm, unsigned int *Xi);

void tm_predict_regression(struct TsetlinMachine *tm, unsigned int *X, int *y, int number_of_examples);

void tm_get_state(struct TsetlinMachine *tm, unsigned int *ta_state);

void tm_set_state(struct TsetlinMachine *tm, unsigned int *ta_state);
