require_relative "test_helper"

class RankerTest < Minitest::Test
  def test_works
    x_train, y_train, x_test, _ = ranker_data
    group = [100, 200]

    model = LightGBM::Ranker.new
    model.fit(x_train, y_train, group: group)
    y_pred = model.predict(x_test)
    expected = [7.996787717548054, 3.431723232058966, 6.839648753163651, 1.044617825216921, 5.539755811471682, 7.546852767597504]
    assert_elements_in_delta expected, y_pred.first(6)

    expected = [198, 181, 170, 83]
    assert_equal expected, model.feature_importances

    model.save_model(tempfile)

    model = LightGBM::Ranker.new
    model.load_model(tempfile)
    assert_equal y_pred, model.predict(x_test)
  end
end
