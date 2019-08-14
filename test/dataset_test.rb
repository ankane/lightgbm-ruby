require_relative "test_helper"

class DatasetTest < Minitest::Test
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
end
