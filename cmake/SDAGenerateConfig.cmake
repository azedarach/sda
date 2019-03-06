function(generate_sda_config_files)

  include(CMakePackageConfigHelpers)

  configure_package_config_file(
    "${CMAKE_MODULE_PATH}/SDAConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/${SDA_INSTALL_CMAKE_DIR}/SDAConfig.cmake"
    INSTALL_DESTINATION "${SDA_INSTALL_CMAKE_DIR}"
    PATH_VARS
    SDA_INSTALL_CMAKE_DIR
    SDA_INSTALL_INCLUDE_DIR
    )

  install(FILES "${PROJECT_BINARY_DIR}/${SDA_INSTALL_CMAKE_DIR}/SDAConfig.cmake"
    DESTINATION ${SDA_INSTALL_CMAKE_DIR})

  install(EXPORT SDATargets
    NAMESPACE SDA::
    DESTINATION ${SDA_INSTALL_CMAKE_DIR})

  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${SDA_INSTALL_CMAKE_DIR}/SDAConfigVersion.cmake"
    VERSION "${SDA_VERSION}"
    COMPATIBILITY SameMajorVersion)

  install(FILES "${PROJECT_BINARY_DIR}/${SDA_INSTALL_CMAKE_DIR}/SDAConfigVersion.cmake"
    DESTINATION ${SDA_INSTALL_CMAKE_DIR})

endfunction()
