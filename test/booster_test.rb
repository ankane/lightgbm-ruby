require_relative "test_helper"

class BoosterTest < Minitest::Test
  def test_predict
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    booster = LightGBM::Booster.new(model_file: "test/support/model.txt")
    y_pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], y_pred.first(2)
  end

  def test_model_str
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    booster = LightGBM::Booster.new(model_str: File.read("test/support/model.txt"))
    y_pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], y_pred.first(2)
  end

  def test_model_from_string
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    booster = LightGBM.train(binary_params, binary_train)
    booster.model_from_string(File.read("test/support/model.txt"))
    y_pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], y_pred.first(2)
  end

  def test_predict_hash
    pred = booster.predict({x0: 3.7, x1: 1.2, x2: 7.2, x3: 9.0})
    assert_in_delta 0.9823112229173586, pred

    pred = booster.predict({"x3" => 9.0, "x2" => 7.2, "x1" => 1.2, "x0" => 3.7})
    assert_in_delta 0.9823112229173586, pred

    pred =
      booster.predict([
        {"x3" => 9.0, "x2" => 7.2, "x1" => 1.2, "x0" => 3.7},
        {"x3" => 0.0, "x2" => 7.9, "x1" => 0.5, "x0" => 7.5}
      ])
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], pred.first(2)

    assert_raises(KeyError) do
      booster.predict({"x0" => 3.7})
    end
  end

  def test_predict_daru
    x_test =
      Daru::DataFrame.new([
        {"x3" => 9.0, "x2" => 7.2, "x1" => 1.2, "x0" => 3.7},
        {"x3" => 0.0, "x2" => 7.9, "x1" => 0.5, "x0" => 7.5}
      ])
    pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], pred.first(2)

    assert_raises(IndexError) do
      booster.predict(Daru::DataFrame.new([{"x0" => 3.7}]))
    end
  end

  def test_predict_rover
    skip if jruby?

    require "rover"
    x_test =
      Rover::DataFrame.new([
        {"x3" => 9.0, "x2" => 7.2, "x1" => 1.2, "x0" => 3.7},
        {"x3" => 0.0, "x2" => 7.9, "x1" => 0.5, "x0" => 7.5}
      ])
    pred = booster.predict(x_test)
    assert_elements_in_delta [0.9823112229173586, 0.9583143724610858], pred.first(2)

    assert_raises(KeyError) do
      booster.predict(Rover::DataFrame.new([{"x0" => 3.7}]))
    end
  end

  def test_predict_type_leaf_index
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    leaf_indexes = booster.predict(x_test, predict_type: LightGBM::C_API_PREDICT_LEAF_INDEX)
    assert_equal 200, leaf_indexes.count
    assert_equal 9.0, leaf_indexes.first
    assert_equal 7.0, leaf_indexes.last

    x_test = [3.7, 1.2, 7.2, 9.0]
    leaf_indexes = booster.predict(x_test, predict_type: LightGBM::C_API_PREDICT_LEAF_INDEX)
    assert_equal 100, leaf_indexes.count
    assert_equal 9.0, leaf_indexes.first
    assert_equal 10.0, leaf_indexes.last
  end

  def test_predict_type_contrib
    x_test = [[3.7, 1.2, 7.2, 9.0], [7.5, 0.5, 7.9, 0.0]]
    results = booster.predict(x_test, predict_type: LightGBM::C_API_PREDICT_CONTRIB)
    assert_equal 10, results.count

    # split results on num_features + 1
    predictions = results.each_slice(5).to_a
    shap_values_1 = predictions.first[0..-2]
    ypred_1 = predictions.first[-1]
    assert_elements_in_delta [
      -0.0733949225678886, -0.24289592050101766, 0.24183795683166504, 0.063430775771174
    ], shap_values_1
    assert_in_delta (0.9933333333834246), ypred_1

    shap_values_2 = predictions.last[0..-2]
    ypred_2 = predictions.last[-1]
    assert_elements_in_delta [
      0.1094902954684793, -0.2810485083947154, 0.26691627597706397, -0.13037702397316747
    ], shap_values_2
    assert_in_delta (0.9933333333834246), ypred_2

    # single row
    x_test = [3.7, 1.2, 7.2, 9.0]
    results = booster.predict(x_test, predict_type: LightGBM::C_API_PREDICT_CONTRIB)
    assert_equal 5, results.count
    shap_values = results[0..-2]
    ypred = results[-1]
    assert_elements_in_delta [
      -0.0733949225678886, -0.24289592050101766, 0.24183795683166504, 0.063430775771174
    ], shap_values
    assert_in_delta (0.9933333333834246), ypred
  end

  def test_feature_importance
    assert_equal [280, 285, 335, 148], booster.feature_importance
  end

  def test_feature_name
    assert_equal ["x0", "x1", "x2", "x3"], booster.feature_name
  end

  def test_feature_importance_bad_importance_type
    error = assert_raises(LightGBM::Error) do
      booster.feature_importance(importance_type: "bad")
    end
    assert_includes error.message, "Unknown importance type"
  end

  def test_model_to_string
    assert booster.model_to_string
  end

  def test_to_json
    assert JSON.parse(booster.to_json)
  end

  def test_dump_model
    assert JSON.parse(booster.dump_model)
  end

  def test_current_iteration
    assert_equal 100, booster.current_iteration
  end

  def test_num_model_per_iteration
    assert_equal 1, booster.num_model_per_iteration
  end

  def test_num_trees
    assert_equal 100, booster.num_trees
  end

  def test_copy
    booster.dup
    booster.clone
  end

  private

  def booster
    @booster ||= LightGBM::Booster.new(model_file: "test/support/model.txt")
  end
end
