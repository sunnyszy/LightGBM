#include <fstream>
#include <string>
#include <iostream>
#include <chrono>
#include <ctime>
#include <unordered_map>
#include <list>
#include <regex>
#include <math.h>
#include <vector>
#include <algorithm>
#include <LightGBM/application.h>

#define HISTFEATURES 50

using namespace std;

struct optEntry {
  uint64_t idx;
  uint64_t volume;
  bool hasNext;

  optEntry(uint64_t idx): idx(idx), volume(numeric_limits<uint64_t>::max()), hasNext(false) {};
};

struct trEntry {
  uint64_t id;
  uint64_t size;
  double cost;
  bool toCache;

  trEntry(uint64_t id, uint64_t size, double cost) : id(id), size(size), cost(cost), toCache(false) {};
};

// cache size tracking
struct cachedObject {
  uint64_t osize;
  bool cached;
};

// from boost hash combine: hashing of std::pairs for unordered_maps
template <class T>
inline void hash_combine(std::size_t & seed, const T & v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

namespace std {
  template<typename S, typename T> struct hash<pair<S, T>> {
    inline size_t operator()(const pair<S, T> & v) const {
      size_t seed = 0;
      ::hash_combine(seed, v.first);
      ::hash_combine(seed, v.second);
      return seed;
    }
  };
}

uint64_t calculateOPT(vector<trEntry> &trace, ifstream &traceFile, uint64_t cacheSize, uint64_t windowSize) {
  uint64_t seq, id, size, idx = 0;
  double cost;

  // from (id, size) to idx
  unordered_map<pair<uint64_t, uint64_t>, uint64_t> lastSeen;
  vector<optEntry> opt;

  while (idx < windowSize && traceFile >> seq >> id >> size >> cost) {
    const auto idsize = make_pair(id, size);
    if (size > 0 && lastSeen.count(idsize) > 0) {
      opt[lastSeen[idsize]].hasNext = true;
      const uint64_t volume = (idx - lastSeen[idsize]) * size;
      opt[lastSeen[idsize]].volume = volume;
    }
    opt.emplace_back(idx);
    trace.emplace_back(id, size, cost);
    lastSeen[idsize] = idx++;
  }
  cout << idx << endl;

  sort(opt.begin(), opt.end(), [](const optEntry& lhs, const optEntry& rhs) {
    return lhs.volume < rhs.volume;
  });

  uint64_t csize = 0;
  for (auto &it: opt) {
    if (csize > cacheSize) {
      break;
    }
    if (it.hasNext) {
      trace[it.idx].toCache = true;
      csize += it.volume / trace.size();
    }
  }
  // whether reach EOF
  return idx;
}

// purpose: derive features and count how many features are inconsistent
void deriveFeatures(vector<trEntry> &trace, const string &path, uint64_t cacheSize) {
  ofstream outfile(path);
  int64_t cacheAvailBytes = cacheSize;
  // from id to intervals
  unordered_map<uint64_t, list<uint64_t> > statistics;
  // from id to object
  unordered_map<uint64_t, cachedObject> cache;
  uint64_t negcachesize = 0;

  uint64_t i = 0;
  for (auto &it: trace) {
    auto &curQueue = statistics[it.id];
    const auto curQueueLen = curQueue.size();
    // drop features larger than 50
    if (curQueueLen > HISTFEATURES) {
      curQueue.pop_back();
    }
    // print features
    if(it.toCache) {
      outfile << 1 << " ";
    } else {
      outfile << 0 << " ";
    }
    size_t idx = 0;
    uint64_t lastReqTime = i;
    for (auto &lit: curQueue) {
      const uint64_t dist = lastReqTime - lit; // distance
      outfile << idx << ":" << dist << " ";
      idx++;
      lastReqTime = lit;
    }

    // object size
    outfile << HISTFEATURES << ":" << round(100.0*log2(it.size)) << " ";

    // update cache size
    uint64_t currentsize;
    if (cacheAvailBytes <= 0) {
      if (cacheAvailBytes < 0) {
        negcachesize++; // that's bad
      }
      currentsize = 0;
    } else {
      currentsize = round(100.0*log2(cacheAvailBytes));
    }
    outfile << HISTFEATURES+1 << ":" << currentsize << " ";
    outfile << HISTFEATURES+2 << ":" << it.cost << "\n";

    if (cache.count(it.id) == 0) {
      // we have never seen this id
      if(it.toCache) {
        cacheAvailBytes -= it.size;
        cache[it.id].cached = true;
      } else {
        cache[it.id].cached = false;
      }
      cache[it.id].osize = it.size;
    } else {
      // repeated request to this id
      if (cache[it.id].cached && !it.toCache) {
        // used to be cached, but not any more
        cacheAvailBytes += cache[it.id].osize;
        cache[it.id].cached = false;
      } else if (!cache[it.id].cached && it.toCache) {
        // wasn't cached, but now it is
        cacheAvailBytes -= it.size;
        cache[it.id].cached = true;
      }
      cache[it.id].osize = it.size;
    }

    // update queue
    curQueue.push_front(i++);
  }

  cerr << "neg. cache size: " << negcachesize << "\n";
  outfile.close();
}

void check(const vector<trEntry> &opt, const string &path, float cutoff, std::ofstream &out) {
  ifstream infile(path);

  bool lmatch;
  double pred;

  uint64_t reqs = 0, matchc = 0, fp = 0, fn = 0, admc = 0;

  for (trEntry dvar : opt) {
    reqs++;
    infile >> pred;

    if (pred <= cutoff) {
      // pred: don't admit
      if (dvar.toCache) {
        // cor: don't admit
        lmatch = false;
        fn++;
        admc++;
      } else {
        lmatch = true;
      }
    } else {
      // pred: admit
      if (dvar.toCache) {
        // cot: admit
        lmatch = true;
        admc++;
      } else {
        lmatch = false;
        fp++;
      }
    }
    if (lmatch) {
      matchc++;
    }
  }

  out << cutoff << " " << reqs << " " << matchc << " " << double(matchc)/reqs << " " << fn << " " << fp << " "
      << admc/double(reqs) << "\n";

  infile.close();
}

void trainModel(const string &trace, const vector<trEntry> &train_opt, const vector<trEntry> &test_opt) {
  const std::string train_trace = "train_" + trace;
  const std::string test_trace = "test_" + trace;
  const std::string model = trace + ".model";
  const std::string train_result = train_trace + ".predict";
  const std::string test_result = test_trace + ".predict";
  const std::string result = trace + ".result";

  std::unordered_map<std::string, std::string> train_params = {
          {"task", "train"},
          {"boosting", "gbdt"},
          {"objective", "binary"},
          {"metric", "binary_logloss,auc"},
          {"metric_freq", "1"},
          {"is_provide_training_metric", "true"},
          {"max_bin", "255"},
          {"num_iterations", "50"},
          {"learning_rate", "0.1"},
          {"num_leaves", "31"},
          {"tree_learner", "serial"},
          {"num_threads", "40"},
          {"feature_fraction", "0.8"},
          {"bagging_freq", "5"},
          {"bagging_fraction", "0.8"},
          {"min_data_in_leaf", "50"},
          {"min_sum_hessian_in_leaf", "5.0"},
          {"is_enable_sparse", "true"},
          {"two_round", "false"},
          {"save_binary", "false"},
          {"data", train_trace},
          {"valid", test_trace},
//          {"input_model", model},
          {"output_model", model}
  };

  std::unordered_map<std::string, std::string> predict_params1 = {
          {"task", "predict"},
          {"data", train_trace},
          {"input_model", model},
          {"output_result", train_result}
  };

  std::unordered_map<std::string, std::string> predict_params2 = {
          {"task", "predict"},
          {"data", test_trace},
          {"input_model", model},
          {"output_result", test_result}
  };

  // train
  LightGBM::Application train_app(train_params);
  train_app.Run();

  // predict
  LightGBM::Application predict_app1(predict_params1);
  predict_app1.Run();
  LightGBM::Application predict_app2(predict_params2);
  predict_app2.Run();

  // test
  std::ofstream out;
  out.open(result);
  out << trace << " train 0.4\n";
  check(train_opt, train_result, 0.4, out);
  out << trace << " test 0.4\n";
  check(test_opt, test_result, 0.4, out);
  out << trace << " train 0.5\n";
  check(train_opt, train_result, 0.5, out);
  out << trace << " test 0.5\n";
  check(test_opt, test_result, 0.5, out);
  out << trace << " train 0.6\n";
  check(train_opt, train_result, 0.6, out);
  out << trace << " test 0.6\n";
  check(test_opt, test_result, 0.6, out);
  out.close();
}

int main(int argc, char* argv[]) {
  if (argc < 4) {
    cerr << "parameters: tracePath cacheSize windowSize\n";
    return 1;
  }

  auto timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
  cout << ctime(&timenow) << endl;
  cout << "start\n";
  // input path
  const string path = argv[1];
  // output name
  string outname;
  const size_t dirSlashIdx = path.rfind('/');
  if (string::npos != dirSlashIdx) {
    outname = path.substr(dirSlashIdx + 1, path.length());
  } else {
    outname = path;
  }
  cout << outname << endl;
  // cache size
  uint64_t cacheSize = stoull(argv[2]);
  // window size
  uint64_t windowSize = stoull(argv[3]);

  ifstream traceFile(path);
  while (true) {
    vector<trEntry> trainTrace;
    vector<trEntry> testTrace;
    uint64_t res = calculateOPT(trainTrace, traceFile, cacheSize, windowSize);
    if (res < windowSize) {
      cerr << "file too short" << endl;
      break;
    }
    calculateOPT(testTrace, traceFile, cacheSize, windowSize);
    // currently still write features to files first, then load data from file for LightGBM
    // TODO: change to create dataset directly from in memory data structures
    // Try ConstructFromSampleData in dataset_loader.h
    deriveFeatures(trainTrace, "train_" + outname, cacheSize);
    deriveFeatures(testTrace, "test_" + outname, cacheSize);
    trainModel(outname, trainTrace, testTrace);
  }

  timenow = chrono::system_clock::to_time_t(chrono::system_clock::now());
  cout << ctime(&timenow) << endl;
  cout << "end\n";

  return 0;
}
