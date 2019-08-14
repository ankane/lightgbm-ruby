require "bundler/setup"
Bundler.require(:default)
require "minitest/autorun"
require "minitest/pride"
require "csv"
require "json"

class Minitest::Test
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
end
