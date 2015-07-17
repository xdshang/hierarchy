#include "data.h"
#include "hierarchical_softmax.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>

NetParam net_param;
DataParam data_param;
Vocab vocab;
DataPair pairs;
LossArg loss_arg;
TrainArg train_arg;
struct SolverParam {
  int num_epoches, num_threads;
  char *syn0_best_file, *syn1_best_file;
  char *syn0_curr_file, *syn1_curr_file;
  char *tree_file;
} solver_param;

void solve() {  
  int a, b, e;
  FILE *fout;
  Dtype best_loss = FLT_MAX, curr_loss, type_loss[PAIR_TYPE_NUM], alpha;
  const Dtype starting_alpha = train_arg.learning_rate;
  pthread_t *pt = (pthread_t*)malloc(solver_param.num_threads * sizeof(pthread_t));
  LossArg* loss_args = (LossArg*)malloc(solver_param.num_threads * sizeof(LossArg));
  TrainArg* train_args = (TrainArg*)malloc(solver_param.num_threads * sizeof(TrainArg));

  printf("Starting training using file %s\n", data_param.train_file);
  printf("Pair size: %lld\n", (long long)pairs.size);

  for (e = 0; e < solver_param.num_epoches; ++e) {
    // Compute loss before update.
    for (a = 0; a < solver_param.num_threads; ++a) {
      memcpy(&loss_args[a], &loss_arg, sizeof(LossArg));
      loss_args[a].thread_id = a;
      pthread_create(&pt[a], NULL, compute_hs_loss, (void*)&loss_args[a]);
    }
    for (a = 0; a < solver_param.num_threads; ++a) {
      pthread_join(pt[a], NULL);
    }
    curr_loss = 0;
    for (b = 0; b < PAIR_TYPE_NUM; ++b) {
      type_loss[b] = 0;
      for (a = 0; a < solver_param.num_threads; ++a) {
        type_loss[b] += loss_args[a].loss[b];
      }
      type_loss[b] /= PAIR_TYPE_NUM * pairs.size;
      curr_loss += type_loss[b];
    }
    printf("Epoch loss: %f,\taa-epoch-loss: %f,\tab-epoch-loss: %f,\tba-epoch-loss: %f,\tbb-epoch-loss: %f\n", 
        curr_loss, type_loss[0], type_loss[1], type_loss[2], type_loss[3]);
    // Save model if necessary.
    if (e % 100 == 0) {
      fout = fopen(solver_param.syn1_curr_file, "wb");
      fprintf(fout, "%lf %d\n", curr_loss, e);
      for (a = 0; a < net_param.vocab->size; ++a) {
        for (b = 0; b < net_param.layer1_size; ++b) {
          fprintf(fout, "%lf ", net_param.syn1[a * net_param.layer1_size + b]);
        }
        fprintf(fout, "\n");
      }
      fclose(fout);
    }
    if (best_loss > curr_loss) {
      best_loss = curr_loss;
      fout = fopen(solver_param.syn1_best_file, "wb");
      fprintf(fout, "%lf %lld %lld\n", best_loss, (long long)net_param.vocab->size, (long long)net_param.layer1_size);
      for (a = 0; a < net_param.vocab->size; ++a) {
        for (b = 0; b < net_param.layer1_size; ++b) {
          fprintf(fout, "%lf ", net_param.syn1[a * net_param.layer1_size + b]);
        }
        fprintf(fout, "\n");
      }
      fclose(fout);
    }
    // Updating.
    alpha = starting_alpha * (1 - (Dtype)e / (solver_param.num_epoches + 1));
    if (alpha < starting_alpha * 0.0001) {
      alpha = starting_alpha * 0.0001;
    }
    printf("Epoch No.: %d\n", e);
    for (a = 0; a < solver_param.num_threads; ++a) {
      memcpy(&train_args[a], &train_arg, sizeof(TrainArg));
      train_args[a].learning_rate = alpha;
      train_args[a].thread_id = a;
      pthread_create(&pt[a], NULL, train_hs, (void*)&train_args[a]);
    }
    for (a = 0; a < solver_param.num_threads; ++a) {
      pthread_join(pt[a], NULL);
    }
  }

  free(pt);
  free(loss_args);
  free(train_args);
}

int arg_pos(const char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  FILE* fout; 

  if ((i = arg_pos("-train", argc, argv)) > 0) {
    data_param.train_file = (char*)malloc((strlen(argv[i + 1]) + 1) * sizeof(char));
    strcpy(data_param.train_file, argv[i + 1]);
  }
  if ((i = arg_pos("-save-vocab", argc, argv)) > 0) {
    data_param.save_vocab_file = (char*)malloc((strlen(argv[i + 1]) + 1) * sizeof(char));
    strcpy(data_param.save_vocab_file, argv[i + 1]);
  }
  if ((i = arg_pos("-read-vocab", argc, argv)) > 0) {
    data_param.read_vocab_file = (char*)malloc((strlen(argv[i + 1]) + 1) * sizeof(char));
    strcpy(data_param.read_vocab_file, argv[i + 1]);
  }
  if ((i = arg_pos("-min-count", argc, argv)) > 0) {
    data_param.min_count = atoi(argv[i + 1]);
  }
  if ((i = arg_pos("-syn0-init", argc, argv)) > 0) {
    net_param.syn0_init_file = (char*)malloc((strlen(argv[i + 1]) + 1) * sizeof(char));
    strcpy(net_param.syn0_init_file, argv[i + 1]);
  }
  if ((i = arg_pos("-size", argc, argv)) > 0) {
    net_param.layer1_size = atoi(argv[i + 1]);
  }
  else {
    printf("Argument 'size' not found, will use the default value: 300.\n");
    net_param.layer1_size = 300;
  }
  if ((i = arg_pos("-learning-rate", argc, argv)) > 0) {
    train_arg.learning_rate = atof(argv[i + 1]);
  }
  else {
    train_arg.learning_rate = 0.001;
  }
  if ((i = arg_pos("-weight-decay", argc, argv)) > 0) {
    train_arg.weight_decay = atof(argv[i + 1]);
  }
  else {
    train_arg.weight_decay = 0.01;
  }
  if ((i = arg_pos("-threads", argc, argv)) > 0) {
    loss_arg.thread_num = atoi(argv[i + 1]);
    train_arg.thread_num = loss_arg.thread_num;
    solver_param.num_threads = loss_arg.thread_num;
  }
  else {
    loss_arg.thread_num = 1;
    train_arg.thread_num = 1;
    solver_param.num_threads = 1;
  }
  if ((i = arg_pos("-epoch", argc, argv)) > 0) {
    solver_param.num_epoches = atoi(argv[i + 1]);
  }
  else {
    solver_param.num_epoches = 1;
  }
  if ((i = arg_pos("-syn0-best", argc, argv)) > 0) {
    solver_param.syn0_best_file = (char*)malloc((strlen(argv[i + 1]) + 1) * sizeof(char));
    strcpy(solver_param.syn0_best_file, argv[i + 1]);
  }
  if ((i = arg_pos("-syn1-best", argc, argv)) > 0) {
    solver_param.syn1_best_file = (char*)malloc((strlen(argv[i + 1]) + 1) * sizeof(char));
    strcpy(solver_param.syn1_best_file, argv[i + 1]);
  }
  if ((i = arg_pos("-syn1-curr", argc, argv)) > 0) {
    solver_param.syn1_curr_file = (char*)malloc((strlen(argv[i + 1]) + 1) * sizeof(char));
    strcpy(solver_param.syn1_curr_file, argv[i + 1]);
  }
  if ((i = arg_pos("-tree-file", argc, argv)) > 0) {
    solver_param.tree_file = (char*)malloc((strlen(argv[i + 1]) + 1) * sizeof(char));
    strcpy(solver_param.tree_file, argv[i + 1]);
  }

  net_param.vocab = &vocab;
  loss_arg.param = &net_param;
  loss_arg.data = &pairs;
  train_arg.param = &net_param;
  train_arg.data = &pairs;
  init_data(&data_param, &vocab, &pairs);
  init_hs(&net_param);

  // save tree to file
  fout = fopen(solver_param.tree_file, "wb");
  for(i = 0; i < vocab.size; ++i) {
    int j;
    fprintf(fout, "%s %d", vocab.data[i].word, (int)vocab.data[i].codelen);
    for(j = 0; j < vocab.data[i].codelen; ++j) {
      fprintf(fout, " %d", (int)vocab.data[i].code[j]);
    }
    for(j = 0; j < vocab.data[i].codelen; ++j) {
      fprintf(fout, " %d", vocab.data[i].point[j]);
    }
    fprintf(fout, "\n");
  }
  fclose(fout);

  solve();
  return 0;
}
