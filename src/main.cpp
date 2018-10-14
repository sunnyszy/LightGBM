#include <iostream>
#include <fstream>
#include <LightGBM/application.h>

void check(const std::string &path1, const std::string &path2, float cutoff, std::ofstream &out) {
  std::ifstream infile1;
  std::ifstream infile2;
  infile1.open(path1);
  infile2.open(path2);

  bool dvar;
  bool lmatch;
  double pred;

  uint64_t reqs = 0, matchc = 0, fp = 0, fn = 0, admc = 0;

  while (!infile1.eof() && !infile2.eof()) {
    reqs++;
    infile1 >> dvar;
    infile2 >> pred;

    if (pred <= cutoff) {
      // pred: don't admit
      if (dvar) {
        // cor: don't admit
        lmatch = false;
        fn++;
        admc++;
      } else {
        lmatch = true;
      }
    } else {
      // pred: admit
      if (dvar) {
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

  infile1.close();
  infile2.close();
}

void test() {
  const std::string train_trace = "../train_dvars_w50mBHR_35.tr";
  const std::string test_trace = "../test_dvars_w50mBHR_35.tr";
  const std::string model = "../dvars_w50mBHR_35.tr.model";
  const std::string train_result = "../train_dvars_w50mBHR_35.tr.predict";
  const std::string test_result = "../test_dvars_w50mBHR_35.tr.predict";
  const std::string train_opt = "../train_dvars_w50mBHR_35.tr.opt";
  const std::string test_opt = "../test_dvars_w50mBHR_35.tr.opt";
  const std::string result = "../result.txt";

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
  out << "dvars_w50mBHR_35.tr train 0.4\n";
  check(train_opt, train_result, 0.4, out);
  out << "dvars_w50mBHR_35.tr test 0.4\n";
  check(test_opt, test_result, 0.4, out);
  out << "dvars_w50mBHR_35.tr train 0.5\n";
  check(train_opt, train_result, 0.5, out);
  out << "dvars_w50mBHR_35.tr test 0.5\n";
  check(test_opt, test_result, 0.5, out);
  out << "dvars_w50mBHR_35.tr train 0.6\n";
  check(train_opt, train_result, 0.6, out);
  out << "dvars_w50mBHR_35.tr test 0.6\n";
  check(test_opt, test_result, 0.6, out);
  out.close();
}

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      test();
    } else {
      LightGBM::Application app(argc, argv);
      app.Run();
    }
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
    exit(-1);
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
    exit(-1);
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
    exit(-1);
  }
}
