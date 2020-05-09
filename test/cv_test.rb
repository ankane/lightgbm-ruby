require_relative "test_helper"

class CvTest < Minitest::Test
  def test_regression
    # need to set stratified=False in Python
    eval_hist = LightGBM.cv(regression_params, regression_train, shuffle: false)
    assert_in_delta 0.2597565400783163, eval_hist["l2-mean"].first
    assert_in_delta 0.10267769399880997, eval_hist["l2-mean"].last
    assert_in_delta 0.07283200245299197, eval_hist["l2-stdv"].first
    assert_in_delta 0.019704697369123978, eval_hist["l2-stdv"].last
  end

  def test_binary
    # need to set stratified=False in Python
    eval_hist = LightGBM.cv(binary_params, binary_train, shuffle: false)
    assert_in_delta 0.3859417693900616, eval_hist["binary_logloss-mean"].first
    assert_in_delta 0.14801378716350452, eval_hist["binary_logloss-mean"].last
    assert_in_delta 0.09986377563273867, eval_hist["binary_logloss-stdv"].first
    assert_in_delta 0.06617990351806904, eval_hist["binary_logloss-stdv"].last
  end

  def test_multiclass
    # need to set stratified=False in Python
    eval_hist = LightGBM.cv(multiclass_params, multiclass_train, shuffle: false)
    assert_in_delta 0.7539432513169948, eval_hist["multi_logloss-mean"].first
    assert_in_delta 0.3505812110913844, eval_hist["multi_logloss-mean"].last
    assert_in_delta 0.11333478721691993, eval_hist["multi_logloss-stdv"].first
    assert_in_delta 0.14174196252890728, eval_hist["multi_logloss-stdv"].last
  end

  def test_early_stopping_early
    eval_hist = nil
    stdout, _ = capture_io do
      eval_hist = LightGBM.cv(regression_params, regression_train, shuffle: false, verbose_eval: true, early_stopping_rounds: 5)
    end
    # TODO replace with 36 for 0.2.0
    assert_equal 41, eval_hist["l2-mean"].size
    assert_includes stdout, "[41]\tcv_agg's l2: 0.0988604 + 0.0243197"
    refute_includes stdout, "[42]"
  end

  def test_early_stopping_not_early
    eval_hist = nil
    stdout, _ = capture_io do
      eval_hist = LightGBM.cv(regression_params, regression_train, shuffle: false, verbose_eval: true, early_stopping_rounds: 500)
    end
    # TODO replace with 36 for 0.2.0
    assert_equal 100, eval_hist["l2-mean"].size
    assert_includes stdout, "[100]\tcv_agg's l2: 0.102678 + 0.0197047"
  end

  def test_multiple_metrics
    params = regression_params.merge(metric: ["l1", "l2", "rmse"])
    eval_hist = LightGBM.cv(params, regression_train, shuffle: false, early_stopping_rounds: 5)
    assert_equal ["l1-mean", "l1-stdv", "l2-mean", "l2-stdv", "rmse-mean", "rmse-stdv"].sort, eval_hist.keys.sort
  end
end
