require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "csv"
require "json"
require "matrix"
require "daru"

class Minitest::Test
  def assert_elements_in_delta(expected, actual)
    assert_equal expected.size, actual.size
    expected.zip(actual) do |exp, act|
      assert_in_delta exp, act
    end
  end

  def regression_data
    @regression_data ||= split_data(*load_data)
  end

  def regression_train
    @regression_train ||= split_train(regression_data)
  end

  def regression_test
    @regression_test ||= split_test(regression_data)
  end

  def binary_data
    x, y = load_data
    y.map! { |v| v > 1 ? 1 : v }
    split_data(x, y)
  end

  def binary_train
    @binary_train ||= split_train(binary_data)
  end

  def binary_test
    @binary_test ||= split_test(binary_data)
  end

  def multiclass_data
    @multiclass_data ||= split_data(*load_data)
  end

  def multiclass_train
    @multiclass_train ||= split_train(multiclass_data)
  end

  def multiclass_test
    @multiclass_test ||= split_test(multiclass_data)
  end

  def ranker_data
    @ranker_data ||= binary_data
  end

  def data_path
    "test/support/data.csv"
  end

  def load_data
    x = []
    CSV.foreach(data_path, headers: true, converters: :numeric) do |row|
      x << row.to_h.values
    end
    y = x.map(&:pop)
    [x, y]
  end

  def split_data(x, y)
    [x[0...300], y[0...300], x[300..-1], y[300..-1]]
  end

  def split_train(data)
    x_train, y_train, _, _ = data
    LightGBM::Dataset.new(x_train, label: y_train)
  end

  def split_test(data)
    _, _, x_test, y_test = data
    LightGBM::Dataset.new(x_test, label: y_test)
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

  def teardown
    @tempfile = nil
  end

  def tempfile
    @tempfile ||= "#{Dir.mktmpdir}/#{Time.now.to_f}"
  end

  def jruby?
    RUBY_PLATFORM == "java"
  end
end
