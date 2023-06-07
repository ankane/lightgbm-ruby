require_relative "test_helper"

class DatasetTest < Minitest::Test
  def test_data_string
    dataset = LightGBM::Dataset.new(data_path, params: {header: true, label_column: "name:y"})
    assert_equal 500, dataset.num_data
    assert_equal 4, dataset.num_feature
    assert_equal 500, dataset.label.size
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

  def test_feature_names
    data = [[1, 2], [3, 4]]
    dataset = LightGBM::Dataset.new(data, feature_names: ["a", "b"])
    assert_equal ["a", "b"], dataset.feature_names
  end

  def test_num_data
    assert_equal 300, regression_train.num_data
  end

  def test_num_feature
    assert_equal 4, regression_train.num_feature
  end

  def test_save_binary
    regression_train.save_binary(tempfile)
    assert File.exist?(tempfile)
  end

  def test_dump_text
    # method is private in Python library
    # https://github.com/microsoft/LightGBM/pull/2434
    assert !regression_train.respond_to?(:dump_text)
    regression_train.send(:dump_text, tempfile)
    assert File.exist?(tempfile)
  end

  def test_matrix
    data = Matrix.build(3, 3) { |row, col| row + col }
    label = Vector.elements([4, 5, 6])
    LightGBM::Dataset.new(data, label: label)
  end

  def test_daru
    data = Daru::DataFrame.from_csv(data_path)
    label = data["y"]
    data = data.delete_vector("y")
    LightGBM::Dataset.new(data, label: label)
  end

  def test_numo
    skip if RUBY_PLATFORM == "java"

    require "numo/narray"
    data = Numo::DFloat.new(3, 5).seq
    label = Numo::DFloat.new(3).seq
    LightGBM::Dataset.new(data, label: label)
  end

  def test_rover
    skip if RUBY_PLATFORM == "java"

    require "rover"
    data = Rover.read_csv(data_path)
    label = data.delete("y")
    LightGBM::Dataset.new(data, label: label)
  end

  def test_copy
    regression_train.dup
    regression_train.clone
  end
end
