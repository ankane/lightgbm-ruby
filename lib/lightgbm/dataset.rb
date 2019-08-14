module LightGBM
  class Dataset
    attr_reader :data, :label

    def initialize(data, label: nil)
      @data = data
      @label = label
    end
  end
end
