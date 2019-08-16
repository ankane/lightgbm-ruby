require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "csv"
require "json"
require "matrix"
require "daru"
require "numo/narray"

class Minitest::Test
  private

  def assert_elements_in_delta(expected, actual)
    assert_equal expected.size, actual.size
    expected.zip(actual) do |exp, act|
      assert_in_delta exp, act
    end
  end

  def boston
    @boston ||= load_csv("boston.csv")
  end

  def boston_train
    @boston_train ||= LightGBM::Dataset.new(boston.data[0...300], label: boston.label[0...300])
  end

  def boston_test
    @boston_test ||= LightGBM::Dataset.new(boston.data[300..-1], label: boston.label[300..-1], reference: boston_train)
  end

  def iris
    @iris ||= load_csv("iris.csv")
  end

  def iris_train
    @iris_train ||= LightGBM::Dataset.new(iris.data[0...100], label: iris.label[0...100])
  end

  def iris_test
    @iris_test ||= LightGBM::Dataset.new(iris.data[100..-1], label: iris.label[100..-1], reference: iris_train)
  end

  def boston_data
    @boston_data ||= begin
      x, y = load_csv("boston.csv", dataset: false)
      [x[0...300], y[0...300], x[300..-1], y[300..-1]]
    end
  end

  def iris_data
    @iris_data ||= begin
      x, y = load_csv("iris.csv", dataset: false)
      [x[0...100], y[0...100], x[100..-1], y[100..-1]]
    end
  end

  def iris_data_binary
    @iris_data_binary ||= begin
      x, y = load_csv("iris.csv", binary: true, dataset: false)
      [x[0...100], y[0...100], x[100..-1], y[100..-1]]
    end
  end

  def load_csv(filename, binary: false, dataset: true)
    x = []
    y = []
    CSV.foreach("test/support/#{filename}", headers: true).each do |row|
      row = row.to_a.map { |_, v| v.to_f }
      x << row[0..-2]
      y << row[-1]
    end
    y = y.map { |v| v > 1 ? 1.0 : v } if binary

    if dataset
      LightGBM::Dataset.new(x, label: y)
    else
      [x, y]
    end
  end

  def regression_params
    {objective: "regression"}
  end

  def binary_params
    {objective: "binary"}
  end

  def multiclass_params
    {objective: "multiclass", num_class: 3}
  end
end
