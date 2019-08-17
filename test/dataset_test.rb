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
    assert_equal 506, boston.num_data
  end

  def test_num_feature
    assert_equal 13, boston.num_feature
  end

  def test_save_binary
    boston.save_binary(tempfile)
    assert File.exist?(tempfile)
  end

  def test_dump_text
    boston.dump_text(tempfile)
    assert File.exist?(tempfile)
  end

  def test_matrix
    data = Matrix.build(3, 3) { |row, col| row + col }
    label = Vector.elements([4, 5, 6])
    LightGBM::Dataset.new(data, label: label)
  end

  def test_daru_data_frame
    data = Daru::DataFrame.from_csv("test/support/boston.csv")
    label = data["medv"]
    data = data.delete_vector("medv")
    LightGBM::Dataset.new(data, label: label)
  end

  def test_numo_narray
    data = Numo::DFloat.new(3, 5).seq
    label = Numo::DFloat.new(3).seq
    LightGBM::Dataset.new(data, label: label)
  end
end
