require_relative "test_helper"

class TrainTest < Minitest::Test
  def test_train
    x_test = test_set.data
    y_test = test_set.label

    params = {objective: "regression", verbosity: -1}
    model = LightGBM.train(params, train_set, valid_sets: [train_set], valid_names: ["train"])
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6

    model.save_model("/tmp/model.txt")
    model = LightGBM::Booster.new(model_file: "/tmp/model.txt")
    y_pred = model.predict(x_test)
    assert_operator rsme(y_test, y_pred), :<=, 6
  end

  def test_cv
    params = {objective: "regression"} #, verbosity: -1}
    model = LightGBM.cv(params, dataset)
  end

  def test_predict
    # LightGBM.train({}, train_set).predict([[20], [50]])
  end

  def test_bad_params
    params = {objective: "regression verbosity=1"}
    assert_raises ArgumentError do
      LightGBM.train(params, train_set)
    end
  end

  # TODO move to dataset test
  def test_num_data
    assert_equal 506, dataset.num_data
  end

  # TODO move to dataset test
  def test_num_feature
    assert_equal 13, dataset.num_feature
  end

  private

  def train_set
    @train_set ||= LightGBM::Dataset.new(dataset.data[0...300], label: dataset.label[0...300])
  end

  def test_set
    @test_set ||= LightGBM::Dataset.new(dataset.data[300..-1], label: dataset.label[300..-1])
  end

  def dataset
    @dataset ||= begin
      x = []
      y = []
      CSV.foreach("test/support/boston.csv", headers: true).each do |row|
        row = row.to_a.map { |_, v| v.to_f }
        x << row[0...13]
        y << row[13]
      end
      LightGBM::Dataset.new(x, label: y)
    end
  end

  def rsme(y_true, y_pred)
    Math.sqrt(y_true.zip(y_pred).map { |a, b| (a - b)**2 }.sum / y_true.size.to_f)
  end
end
