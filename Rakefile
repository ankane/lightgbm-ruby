require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = false
end

shared_libraries = %w(lib_lightgbm.dll lib_lightgbm.dylib lib_lightgbm.arm64.dylib lib_lightgbm.so)

# ensure vendor files exist
task :ensure_vendor do
  shared_libraries.each do |file|
    raise "Missing file: #{file}" unless File.exist?("vendor/#{file}")
  end
end

Rake::Task["build"].enhance [:ensure_vendor]

def download_file(file, sha256)
  require "open-uri"

  version = "3.3.0"

  url =
    if file == "lib_lightgbm.arm64.dylib"
      "https://github.com/ankane/ml-builds/releases/download/lightgbm-#{version}/#{file}"
    else
      "https://github.com/microsoft/LightGBM/releases/download/v#{version}/#{file}"
    end
  puts "Downloading #{file}..."
  contents = URI.open(url).read

  computed_sha256 = Digest::SHA256.hexdigest(contents)
  raise "Bad hash: #{computed_sha256}" if computed_sha256 != sha256

  dest = "vendor/#{file}"
  File.binwrite(dest, contents)
  puts "Saved #{dest}"
end

# https://github.com/microsoft/LightGBM/releases
namespace :vendor do
  task :linux do
    download_file("lib_lightgbm.so", "6f21ea27f6f50a1631bd58e51d69efe32cb27b0f3e95643ed4a6996bca06cb1b")
  end

  task :mac do
    download_file("lib_lightgbm.dylib", "d353459a6b3dda59ba03ba99f03b9087109d398ddede870d097b6353b0dd5961")
    download_file("lib_lightgbm.arm64.dylib", "fdf47280c818d1a18eb1c7a85152281f2a12f4fbe16888de5d1d535f572f011b")
  end

  task :windows do
    download_file("lib_lightgbm.dll", "df983d5cf24956b5689c4bf4ed6a902cbc9ac45cfac687fdafcd222c9e5d2d71")
  end

  task all: [:linux, :mac, :windows]

  task :platform do
    if Gem.win_platform?
      Rake::Task["vendor:windows"].invoke
    elsif RbConfig::CONFIG["host_os"] =~ /darwin/i
      Rake::Task["vendor:mac"].invoke
    else
      Rake::Task["vendor:linux"].invoke
    end
  end
end
