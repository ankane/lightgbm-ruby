require "bundler/gem_tasks"
require "rake/testtask"

task default: :test
Rake::TestTask.new do |t|
  t.libs << "test"
  t.pattern = "test/**/*_test.rb"
  t.warning = false # for daru
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

  # also update licenses in vendor/
  version = "4.0.0"

  url =
    if file == "lib_lightgbm.arm64.dylib"
      "https://github.com/ankane/ml-builds/releases/download/lightgbm-#{version}/#{file}"
    else
      "https://github.com/microsoft/LightGBM/releases/download/v#{version}/#{file}"
    end
  puts "Downloading #{file}..."
  contents = URI.parse(url).read

  computed_sha256 = Digest::SHA256.hexdigest(contents)
  raise "Bad hash: #{computed_sha256}" if computed_sha256 != sha256

  dest = "vendor/#{file}"
  File.binwrite(dest, contents)
  puts "Saved #{dest}"
end

# https://github.com/microsoft/LightGBM/releases
namespace :vendor do
  task :linux do
    download_file("lib_lightgbm.so", "97e8f4fda11f9546aac6a00b9f787754c372f5f2b1668b45c4f3b6f17dc7a855")
  end

  task :mac do
    download_file("lib_lightgbm.dylib", "d4415026f3c02ac40c1b85e87646c70cf0da8c6bb6446a24c01b6a5cd58eee7e")
    download_file("lib_lightgbm.arm64.dylib", "6f00e3b734ed5f5495d8837324c4ef717b858b7205267c1b1c9b43d8b0bd823a")
  end

  task :windows do
    download_file("lib_lightgbm.dll", "52fbbdc0ccd4cfac9e5f7340cc349b93810f59d4791ec3d91dd8ac9fbe55197f")
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
