

#include "ConvolutionalTsetlinMachine.h"

struct MultiClassTsetlinMachine {
	int number_of_classes;
	struct TsetlinMachine **tsetlin_machines;

	int number_of_patches;
	int number_of_ta_chunks;
	int number_of_state_bits;
};

struct MultiClassTsetlinMachine *CreateMultiClassTsetlinMachine(int number_of_classes, int number_of_clauses, int number_of_features, int number_of_patches, int number_of_ta_chunks, int number_of_state_bits, int T, double s, double s_range, int boost_true_positive_feedback, int weighted_clauses);

void mc_tm_initialize(struct MultiClassTsetlinMachine *mc_tm);

void mc_tm_destroy(struct MultiClassTsetlinMachine *mc_tm);

void mc_tm_initialize_random_streams(struct MultiClassTsetlinMachine *mc_tm, float s);

void mc_tm_predict(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int *y, int number_of_examples);

void mc_tm_fit(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X, int y[], int number_of_examples, int epochs);

void mc_tm_get_state(struct MultiClassTsetlinMachine *mc_tm, int class, unsigned int *ta_state);

void mc_tm_set_state(struct MultiClassTsetlinMachine *mc_tm, int class, unsigned int *ta_state);

int mc_tm_ta_state(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, int ta);

int mc_tm_ta_action(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, int ta);

void mc_tm_transform(struct MultiClassTsetlinMachine *mc_tm, unsigned int *X,  unsigned int *X_transformed, int invert, int number_of_examples);

void mc_tm_clause_configuration(struct MultiClassTsetlinMachine *mc_tm, int class, int clause, unsigned int *clause_configuration);

