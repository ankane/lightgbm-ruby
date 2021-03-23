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

  version = "3.2.0"

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
    download_file("lib_lightgbm.so", "f46ba293e4312f65282509267f7374736e5134dd5f0ef6692359a98d273d6b55")
  end

  task :mac do
    download_file("lib_lightgbm.dylib", "d959c65218851252ec5a0aa89c031cc7e94600e95c88f7b1a4392a1c38f3b18a")
    download_file("lib_lightgbm.arm64.dylib", "e3aeac2c66cbc5d5870ca4c02c5468e055f0a36fae5eeec95ecffb57be81cea0")
  end

  task :windows do
    download_file("lib_lightgbm.dll", "746f641504e6d8f165568d70d85a659b2411294f18e86f22ca4c3db64357432f")
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
