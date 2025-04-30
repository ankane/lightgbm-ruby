require_relative "test_helper"

class ClassifierTest < Minitest::Test
  def test_binary
    x_train, y_train, x_test, _ = binary_data

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    y_pred_proba = model.predict_proba(x_test)
    expected = [9.243317488749625e-06, 0.9999907566825113]
    assert_elements_in_delta expected, y_pred_proba.first

    expected = [399, 367, 419, 140]
    assert_equal expected, model.feature_importances

    model.save_model(tempfile)

    model = LightGBM::Classifier.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_multiclass
    x_train, y_train, x_test, _ = multiclass_data

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    y_pred_proba = model.predict_proba(x_test)
    expected = [0.00036627031584163575, 0.9456350323547973, 0.053998697329361176]
    assert_elements_in_delta expected, y_pred_proba.first

    expected = [1118, 1060, 1272, 441]
    assert_equal expected, model.feature_importances

    model.save_model(tempfile)

    model = LightGBM::Classifier.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_early_stopping
    x_train, y_train, x_test, y_test = multiclass_data

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train, early_stopping_rounds: 5, eval_set: [[x_test, y_test]], verbose: false)
    assert_equal 54, model.best_iteration
  end

  def test_missing_numeric
    x_train, y_train, x_test, _ = multiclass_data

    x_train = x_train.map(&:dup)
    x_test = x_test.map(&:dup)
    [x_train, x_test].each do |xt|
      xt.each do |x|
        x.size.times do |i|
          x[i] = nil if x[i] == 3.7
        end
      end
    end

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    expected = [1, 2, 2, 1, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    expected = [1140, 1046, 1309, 427]
    assert_equal expected, model.feature_importances
  end

  def test_missing_categorical
    x_train, y_train, x_test, _ = multiclass_data

    x_train = x_train.map(&:dup)
    x_test = x_test.map(&:dup)
    [x_train, x_test].each do |xt|
      xt.each do |x|
        x[3] = nil if x[3] > 7
      end
    end

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train, categorical_feature: [3])

    y_pred = model.predict(x_test)
    expected = [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    expected = [1228, 1265, 1446, 30]
    assert_equal expected, model.feature_importances
  end
end
