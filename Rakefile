require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = false
end

# https://github.com/microsoft/LightGBM/releases
task :vendor do
  require "open-uri"

  version = "2.3.0"

  %w(lib_lightgbm.dll lib_lightgbm.dylib lib_lightgbm.so).each do |file|
    url = "https://github.com/microsoft/LightGBM/releases/download/v#{version}/#{file}"
    puts "Saving #{file}..."
    File.binwrite("vendor/#{file}", open(url).read)
  end
end
