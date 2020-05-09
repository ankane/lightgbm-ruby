require_relative "test_helper"

class ClassifierTest < Minitest::Test
  def test_binary
    x_train, y_train, x_test, _ = binary_data

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    y_pred_proba = model.predict_proba(x_test)
    expected = [1.0329792027752305e-05, 0.9999896702079722]
    assert_elements_in_delta expected, y_pred_proba.first

    expected = [338, 281, 241, 174]
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
    expected = [1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 2, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    y_pred_proba = model.predict_proba(x_test)
    expected = [0.0004993587611819779, 0.9439989811698228, 0.05550166006899516]
    assert_elements_in_delta expected, y_pred_proba.first

    expected = [826, 912, 1093, 398]
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
    assert_equal 78, model.best_iteration
  end

  def test_missing_numeric
    x_train, y_train, x_test, _ = multiclass_data

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
    expected = [1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 2, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    expected = [834, 945, 1069, 389]
    assert_equal expected, model.feature_importances
  end

  def test_missing_categorical
    x_train, y_train, x_test, _ = multiclass_data

    [x_train, x_test].each do |xt|
      xt.each do |x|
        x[3] = nil if x[3] > 7
      end
    end

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train, categorical_feature: [3])

    y_pred = model.predict(x_test)
    expected = [1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 2, 1, 1, 1, 1, 0, 1, 1, 1, 2, 2, 1, 1, 1, 0, 1, 2, 1, 1, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1]
    assert_equal expected, y_pred.first(100)

    expected = [1002, 1025, 1126, 80]
    assert_equal expected, model.feature_importances
  end
end
