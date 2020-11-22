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

Rake::Task["build"].enhance [:ensure_vendor]

def download_file(file)
  require "open-uri"

  version = "3.1.0"

  url = "https://github.com/ankane/ml-builds/releases/download/lightgbm-#{version}/#{file}"
  puts "Downloading #{file}..."
  dest = "vendor/#{file}"
  File.binwrite(dest, URI.open(url).read)
  puts "Saved #{dest}"
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
