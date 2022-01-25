require_relative "lib/lightgbm/version"

Gem::Specification.new do |spec|
  spec.name          = "lightgbm"
  spec.version       = LightGBM::VERSION
  spec.summary       = "High performance gradient boosting for Ruby"
  spec.homepage      = "https://github.com/ruby-ml/lightgbm-ruby"
  spec.license       = "MIT"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@ankane.org"

  spec.files         = Dir["*.{md,txt}", "{lib,vendor}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.4"

  spec.add_dependency "ffi"
end
