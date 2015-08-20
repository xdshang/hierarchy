#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <pthread.h>
#include <assert.h>

#include "data.h"
#include "hierarchical_softmax.h"

NetParam net_param;
DataParam data_param;
Vocab vocab;
DataPair pairs;
LossArg loss_arg;
TrainArg train_arg;
struct SolverParam {
  int num_epoches, num_threads, batch_size;
  char *syn0_best_file, *syn1_best_file;
  char *syn0_curr_file, *syn1_curr_file;
  char *tree_file;
} solver_param;

void _generate_fet_list(const int id, int* size0, int* size1, 
    int* list0, int* list1) {
  int bstart = id * solver_param.batch_size;
  int bend = (id + 1) * solver_param.batch_size;
  int i, j, p;

  if (bend > pairs.size) {
    bend = pairs.size;
  }
  *size0 = 0;
  *size1 = 0;
  for (i = bstart; i < bend; ++i) {
    p = search_vocab(&vocab, pairs.data[i].first);
    list0[(*size0)++] = p;
    for (j = 0; j < vocab.data[p].codelen; ++j) {
      list1[(*size1)++] = vocab.data[p].point[j];
    }
    p = search_vocab(&vocab, pairs.data[i].second);
    list0[(*size0)++] = p;
    for (j = 0; j < vocab.data[p].codelen; ++j) {
      list1[(*size1)++] = vocab.data[p].point[j];
    }        
  }  
}

void _compute_boundary(const int total_size, const int batch_size,
    const int thread_size, const int batch_id, const int thread_id, 
    int* pstart, int* pend) {
  *pstart = batch_id * batch_size + thread_id * thread_size;
  *pend = (thread_id + 1) * thread_size;
  if (*pend > batch_size) {
    *pend = batch_size;
  }
  *pend += batch_id * batch_size;
  if (*pend > total_size) {
    *pend = total_size;
  }
}

void solve() {
  const int num_batches = (pairs.size - 1) / solver_param.batch_size + 1;
  const int thread_size = (solver_param.batch_size - 1) / solver_param.num_threads + 1;
  int a, b, e, i;
  Dtype curr_loss, type_loss[PAIR_TYPE_NUM], alpha;
  const Dtype starting_alpha = train_arg.learning_rate;
  pthread_t fet0_thread, fet1_thread, pt[solver_param.num_threads];
  LossArg loss_args[solver_param.num_threads];
  TrainArg train_args[solver_param.num_threads];
  int fet0_list[MAX_NUM_LIST * MAX_NUM_LIST], fet1_list[MAX_NUM_LIST * MAX_NUM_LIST];
  PrefetchArgs fet0_args, fet1_args;

  if (solver_param.batch_size > MAX_NUM_LIST) {
    printf("Error: batch size is too large for MAX_NUM_LIST.\n");
    return;
  }

  fet0_args.hid = net_param.syn0_hid;
  fet0_args.list = fet0_list;
  fet1_args.hid = net_param.syn1_hid;
  fet1_args.list = fet1_list;

  printf("Starting training using file %s\n", data_param.train_file);
  printf("Pair size: %d\n", pairs.size);
  for (e = 0; e < solver_param.num_epoches; ++e) {
    // Compute loss.
    for (b = 0; b < PAIR_TYPE_NUM; ++b) {
      type_loss[b] = 0;
    }
    // fetch the 1st batch
    _generate_fet_list(0, &(fet0_args.size), &(fet1_args.size),
        fet0_args.list, fet1_args.list);
    pthread_create(&fet0_thread, NULL, prefetch_sync_param, (void*)&fet0_args);
    pthread_create(&fet1_thread, NULL, prefetch_sync_param, (void*)&fet1_args);
    pthread_join(fet0_thread, NULL);
    pthread_join(fet1_thread, NULL);
    assert(fet0_args.status == 0);
    assert(fet1_args.status == 0);
    // iterate over batches
    for (i = 0; i < num_batches; ++i) {
      _generate_fet_list(i + 1, &(fet0_args.size), &(fet1_args.size),
          fet0_args.list, fet1_args.list);
      pthread_create(&fet0_thread, NULL, prefetch_sync_param, (void*)&fet0_args);
      pthread_create(&fet1_thread, NULL, prefetch_sync_param, (void*)&fet1_args);
      for (a = 0; a < solver_param.num_threads; ++a) {
        memcpy(&loss_args[a], &loss_arg, sizeof(LossArg));
        _compute_boundary(pairs.size, solver_param.batch_size, thread_size,
            i, a, &(loss_args[a].pstart), &(loss_args[a].pend));
        pthread_create(&pt[a], NULL, compute_hs_loss, (void*)&loss_args[a]);
      }
      for (a = 0; a < solver_param.num_threads; ++a) {
        pthread_join(pt[a], NULL);
      }
      for (b = 0; b < PAIR_TYPE_NUM; ++b) {
        for (a = 0; a < solver_param.num_threads; ++a) {
          type_loss[b] += loss_args[a].loss[b];
        }
      }
      pthread_join(fet0_thread, NULL);
      pthread_join(fet1_thread, NULL);
      assert(fet0_args.status == 0);
      assert(fet1_args.status == 0);
    }
    curr_loss = 0;
    for (b = 0; b < PAIR_TYPE_NUM; ++b) {
      type_loss[b] /= PAIR_TYPE_NUM * pairs.size;
      curr_loss += type_loss[b];
    }
    printf("Epoch No.: %d, \tloss: %f\n\taa-loss: %f,\tab-loss: %f,\tba-loss: %f,\tbb-loss: %f\n", 
        e, curr_loss, type_loss[0], type_loss[1], type_loss[2], type_loss[3]);

    // Updating.
    alpha = starting_alpha;
    // alpha = starting_alpha * (1 - (Dtype)e / (solver_param.num_epoches + 1));
    // if (alpha < starting_alpha * 0.0001) {
    //   alpha = starting_alpha * 0.0001;
    // }
    // prefetch the first batch
    _generate_fet_list(0, &(fet0_args.size), &(fet1_args.size),
        fet0_args.list, fet1_args.list);
    pthread_create(&fet0_thread, NULL, prefetch_sync_param, (void*)&fet0_args);
    pthread_create(&fet1_thread, NULL, prefetch_sync_param, (void*)&fet1_args);
    pthread_join(fet0_thread, NULL);
    pthread_join(fet1_thread, NULL);
    assert(fet0_args.status == 0);
    assert(fet1_args.status == 0);
    for (i = 0; i < num_batches; ++i) {
      // lanch a new thread to fetch next batch
      _generate_fet_list(i + 1, &(fet0_args.size), &(fet1_args.size),
          fet0_args.list, fet1_args.list);
      pthread_create(&fet0_thread, NULL, prefetch_sync_param, (void*)&fet0_args);
      pthread_create(&fet1_thread, NULL, prefetch_sync_param, (void*)&fet1_args);      
      for (a = 0; a < solver_param.num_threads; ++a) {
        memcpy(&train_args[a], &train_arg, sizeof(TrainArg));
        _compute_boundary(pairs.size, solver_param.batch_size, thread_size,
            i, a, &(train_args[a].pstart), &(train_args[a].pend));
        train_args[a].learning_rate = alpha;
        pthread_create(&pt[a], NULL, train_hs, (void*)&train_args[a]);
      }
      for (a = 0; a < solver_param.num_threads; ++a) {
        pthread_join(pt[a], NULL);
      }
      // wait for the prefetching thread
      pthread_join(fet0_thread, NULL);
      pthread_join(fet1_thread, NULL);
      assert(fet0_args.status == 0);
      assert(fet1_args.status == 0);
    }
  }
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
  if ((i = arg_pos("-syn1-init", argc, argv)) > 0) {
    net_param.syn1_init_file = (char*)malloc((strlen(argv[i + 1]) + 1) * sizeof(char));
    strcpy(net_param.syn1_init_file, argv[i + 1]);
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
  if ((i = arg_pos("-batch-size", argc, argv)) > 0) {
    solver_param.batch_size = atoi(argv[i + 1]);
  }
  else {
    solver_param.batch_size = MAX_NUM_LIST;
  }
  if ((i = arg_pos("-threads", argc, argv)) > 0) {
    solver_param.num_threads = atoi(argv[i + 1]);
  }
  else {
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
  // TODO: shoud be handled by the model
  assert(destroy_sync_param(net_param.syn0_hid) == 0);
  assert(destroy_sync_param(net_param.syn1_hid) == 0);

  return 0;
}
