require_relative "test_helper"

class LightGBMTest < Minitest::Test
  def test_load_model
    x_test = [
      [0.04417, 70.0, 2.24, 0, 0.400, 6.871, 47.4, 7.8278, 5, 358, 14.8, 390.86, 6.07],
      [0.03537, 34.0, 6.09, 0, 0.433, 6.590, 40.4, 5.4917, 7, 329, 16.1, 395.75, 9.50]
    ]

    model = LightGBM::Booster.new(model_file: "test/support/model.txt")
    y_pred = model.predict(x_test)
    assert_in_delta 28.29122797, y_pred[0]
    assert_in_delta 25.87936514, y_pred[1]
  end

  def test_train
    x = []
    y = []
    CSV.foreach("test/support/boston.csv", headers: true).each do |row|
      row = row.to_a.map { |_, v| v.to_f }
      x << row[0...13]
      y << row[13]
    end
    x_train = x[0...300]
    y_train = y[0...300]
    x_test = x[300..-1]
    y_test = y[300..-1]

    params = {objective: "regression", verbosity: -1}
    train_set = LightGBM::Dataset.new(x_train, label: y_train)
    model = LightGBM.train(params, train_set)
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6

    model.save_model("/tmp/model.txt")
    model = LightGBM::Booster.new(model_file: "/tmp/model.txt")
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6
  end

  private

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
