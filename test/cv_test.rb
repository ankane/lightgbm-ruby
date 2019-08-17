require_relative "test_helper"

class CvTest < Minitest::Test
  def test_regression
    eval_hist = LightGBM.cv(regression_params, boston, shuffle: false)
    assert_in_delta 82.33637413467392, eval_hist["l2-mean"].first
    assert_in_delta 22.55870116923647, eval_hist["l2-mean"].last
    assert_in_delta 35.018415375374886, eval_hist["l2-stdv"].first
    assert_in_delta 11.605523321472438, eval_hist["l2-stdv"].last
  end

  def test_binary
    # need to set stratified=False in Python
    eval_hist = LightGBM.cv(binary_params, iris, shuffle: false)
    assert_in_delta 0.5523814945253853, eval_hist["binary_logloss-mean"].first
    assert_in_delta 0.0702413393927758, eval_hist["binary_logloss-mean"].last
    assert_in_delta 0.04849276982520402, eval_hist["binary_logloss-stdv"].first
    assert_in_delta 0.14004060158158324, eval_hist["binary_logloss-stdv"].last
  end

  def test_multiclass
    # need to set stratified=False in Python
    eval_hist = LightGBM.cv(multiclass_params, iris, shuffle: false)
    assert_in_delta 0.9968127754694314, eval_hist["multi_logloss-mean"].first
    assert_in_delta 0.23619145913652034, eval_hist["multi_logloss-mean"].last
    assert_in_delta 0.017988535469258864, eval_hist["multi_logloss-stdv"].first
    assert_in_delta 0.19730616941199997, eval_hist["multi_logloss-stdv"].last
  end

  def test_early_stopping_early
    eval_hist = nil
    stdout, _ = capture_io do
      eval_hist = LightGBM.cv(regression_params, boston, shuffle: false, verbose_eval: true, early_stopping_rounds: 5)
    end
    assert_equal 49, eval_hist["l2-mean"].size
    assert_includes stdout, "[49]\tcv_agg's l2: 21.6348 + 12.0872"
    refute_includes stdout, "[50]"
  end

  def test_early_stopping_not_early
    eval_hist = nil
    stdout, _ = capture_io do
      eval_hist = LightGBM.cv(regression_params, boston, shuffle: false, verbose_eval: true, early_stopping_rounds: 500)
    end
    assert_equal 100, eval_hist["l2-mean"].size
    assert_includes stdout, "[100]\tcv_agg's l2: 22.5587 + 11.6055"
  end

  def test_multiple_metrics
    params = regression_params.dup
    params[:metric] = ["l1", "l2", "rmse"]
    eval_hist = LightGBM.cv(params, boston, shuffle: false, early_stopping_rounds: 5)
    assert_equal ["l1-mean", "l1-stdv", "l2-mean", "l2-stdv", "rmse-mean", "rmse-stdv"].sort, eval_hist.keys.sort
  end
end
