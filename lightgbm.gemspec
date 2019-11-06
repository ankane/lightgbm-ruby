require_relative "lib/lightgbm/version"

Gem::Specification.new do |spec|
  spec.name          = "lightgbm"
  spec.version       = LightGBM::VERSION
  spec.summary       = "LightGBM - the high performance machine learning library - for Ruby"
  spec.homepage      = "https://github.com/ankane/lightgbm"
  spec.license       = "MIT"

  spec.author        = "Andrew Kane"
  spec.email         = "andrew@chartkick.com"

  spec.files         = Dir["*.{md,txt}", "{lib,vendor}/**/*"]
  spec.require_path  = "lib"

  spec.required_ruby_version = ">= 2.4"

  spec.add_development_dependency "bundler"
  spec.add_development_dependency "rake"
  spec.add_development_dependency "minitest", ">= 5"
  spec.add_development_dependency "daru"
end
