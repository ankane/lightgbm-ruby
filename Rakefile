require "bundler/gem_tasks"
require "rake/testtask"
require "ruby_memcheck"

test_config = lambda do |t|
  t.pattern = "test/**/*_test.rb"
end
Rake::TestTask.new(&test_config)

namespace :test do
  RubyMemcheck::TestTask.new(:valgrind, &test_config)
end

task default: :test

shared_libraries = %w(lib_lightgbm.dll lib_lightgbm.dylib lib_lightgbm.arm64.dylib lib_lightgbm.so lib_lightgbm.arm64.so)

# ensure vendor files exist
task :ensure_vendor do
  shared_libraries.each do |file|
    raise "Missing file: #{file}" unless File.exist?("vendor/#{file}")
  end
end

Rake::Task["build"].enhance [:ensure_vendor]

def download_file(file, sha256)
  require "open-uri"

  # also update licenses in vendor/
  version = "4.7.0"

  url =
    if ["lib_lightgbm.arm64.so", "lib_lightgbm.arm64.dylib"].include?(file)
      "https://github.com/ankane/ml-builds/releases/download/lightgbm-#{version}/#{file}"
    else
      "https://github.com/lightgbm-org/LightGBM/releases/download/v#{version}/#{file}"
    end
  puts "Downloading #{file}..."
  contents = URI.parse(url).read

  computed_sha256 = Digest::SHA256.hexdigest(contents)
  raise "Bad hash: #{computed_sha256}" if computed_sha256 != sha256

  dest = "vendor/#{file}"
  File.binwrite(dest, contents)
  puts "Saved #{dest}"
end

# https://github.com/lightgbm-org/LightGBM/releases
namespace :vendor do
  task :linux do
    download_file("lib_lightgbm.so", "38c642ab2ac37312e414ce4c723358ad48f0cd51703b5c7ca30f96bdd131e21d")
    download_file("lib_lightgbm.arm64.so", "3b2e6a5cf8f71a428ca14122349a160c0cc087955d98affeaf28aea70858977f")
  end

  task :mac do
    download_file("lib_lightgbm.dylib", "e4b2225fa7a946894be28422d4f53cdc7d2e5f195ff236ea1b012a1db2949345")
    download_file("lib_lightgbm.arm64.dylib", "8b2027a92000f8f92622259cd2dfe614cfb96561126fcdcf9636887ee64b0bbb")
  end

  task :windows do
    download_file("lib_lightgbm.dll", "c89524b9913016a2f07bcfd795ba931b93c2a97b2775aa903cc3ef8ae1571834")
  end

  task all: [:linux, :mac, :windows]

  task :platform do
    if Gem.win_platform?
      Rake::Task["vendor:windows"].invoke
    elsif RbConfig::CONFIG["host_os"].match?(/darwin/i)
      Rake::Task["vendor:mac"].invoke
    else
      Rake::Task["vendor:linux"].invoke
    end
  end
end
