require_relative "test_helper"

class DatasetTest < Minitest::Test
  def test_data_string
    dataset = LightGBM::Dataset.new("test/support/boston.csv", params: {header: true, label_column: "name:medv"})
    assert_equal 506, dataset.num_data
    assert_equal 13, dataset.num_feature
    assert_equal 506, dataset.label.size
  end

  def test_label
    data = [[1, 2], [3, 4]]
    label = [1, 2]
    dataset = LightGBM::Dataset.new(data, label: label)
    assert label, dataset.label
  end

  def test_weight
    data = [[1, 2], [3, 4]]
    weight = [1, 2]
    dataset = LightGBM::Dataset.new(data, weight: weight)
    assert weight, dataset.weight
  end

  def test_num_data
    assert_equal 506, dataset.num_data
  end

  def test_num_feature
    assert_equal 13, dataset.num_feature
  end

  def test_save_binary
    dataset.save_binary("/tmp/train.bin")
    assert File.exist?("/tmp/train.bin")
  end
end
