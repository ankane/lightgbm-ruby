require_relative "test_helper"

class ClassifierTest < Minitest::Test
  def test_binary
    x_train, y_train, x_test, _ = iris_data_binary

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
    assert_equal expected, y_pred

    model.save_model(tempfile)

    model = LightGBM::Classifier.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_multiclass
    x_train, y_train, x_test, _ = iris_data

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    expected = [2, 2, 0, 1, 1, 1, 1, 2, 1, 1, 0, 1, 0, 2, 1, 1, 2, 2, 1, 1, 1, 0, 0, 1, 1, 2, 1, 2, 0, 2, 1, 1, 2, 2, 2, 1, 0, 2, 2, 1, 1, 1, 1, 0, 1, 2, 0, 2, 1, 1]

    assert_equal expected, y_pred

    model.save_model(tempfile)

    model = LightGBM::Classifier.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end

  def test_predict_proba_binary
    x_train, y_train, x_test, _ = iris_data_binary

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)

    y_pred = model.predict_proba(x_test)
    expected = [1.63425933e-05, 9.99983657e-01]
    assert_elements_in_delta expected, y_pred[0]
  end

  def test_predict_proba_multiclass
    x_train, y_train, x_test, _ = iris_data

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)

    y_pred = model.predict_proba(x_test)
    expected = [3.91608299e-04, 3.81933551e-01, 6.17674841e-01]
    assert_elements_in_delta expected, y_pred[0]
  end

  def test_feature_importances_binary
    x_train, y_train, _, _ = iris_data_binary

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)

    expected = [28, 112, 94, 62]
    assert_equal expected, model.feature_importances
  end

  def test_feature_importances_multiclass
    x_train, y_train, _, _ = iris_data

    model = LightGBM::Classifier.new
    model.fit(x_train, y_train)

    expected = [93, 281, 277, 220]
    assert_equal expected, model.feature_importances
  end
end
