require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = false
end

shared_libraries = %w(lib_lightgbm.dll lib_lightgbm.dylib lib_lightgbm.so)

# ensure vendor files exist
task :ensure_vendor do
  shared_libraries.each do |file|
    raise "Missing file: #{file}" unless File.exist?("vendor/#{file}")
  end
end

Rake::Task["release:guard_clean"].enhance [:ensure_vendor]

def download_file(file)
  require "open-uri"

  version = "2.3.1"

  url = "https://github.com/microsoft/LightGBM/releases/download/v#{version}/#{file}"
  puts "Downloading #{file}..."
  File.binwrite("vendor/#{file}", open(url).read)
end

# https://github.com/microsoft/LightGBM/releases
namespace :vendor do
  task :linux do
    download_file("lib_lightgbm.so")
  end

  task :mac do
    download_file("lib_lightgbm.dylib")
  end

  task :windows do
    download_file("lib_lightgbm.dll")
  end

  task all: [:linux, :mac, :windows]
end
