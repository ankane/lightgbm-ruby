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
  version = "4.5.0"

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
    download_file("lib_lightgbm.so", "4b2b68c4d0fa99bace6cc540224b457ff899ccee0fdc8875e4625a38b00fc5e5")
  end

  task :mac do
    download_file("lib_lightgbm.dylib", "b02d48071ba4ae1e13e336a902dc5f82a5732de4448d47a20d8e9d94d5d3db2a")
    download_file("lib_lightgbm.arm64.dylib", "840e16754db0d3e4852bdfdecc1ee08bc367b138e0bf18fabb4ce3d9b39c936a")
  end

  task :windows do
    download_file("lib_lightgbm.dll", "1d281ec96684806d83468469fb6052880308f39bf03a34d85ee9aa38195d260c")
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
