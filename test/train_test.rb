require_relative "test_helper"

class TrainTest < Minitest::Test
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
    model = LightGBM.train(params, train_set, valid_sets: [train_set], valid_names: ["train"])
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
